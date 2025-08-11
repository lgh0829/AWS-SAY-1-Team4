-- 안전한 로깅
local function SafeLog(message)
  if _G.PrintLog ~= nil then PrintLog(message)
  elseif _G.print ~= nil then print(message) end
end

-- GET 래퍼
local function SafeRestApiGet(path)
  local ok, res = pcall(function() return RestApiGet(path) end)
  if not ok then
    SafeLog('[lua] RestApiGet failed: ' .. path .. ' err=' .. tostring(res))
    return nil
  end
  return res
end

-- Post 래퍼 (path 기반)
local function SafeRestApiPost(path, body, headers)
  local ok, res = pcall(function()
    return RestApiPost(path, body or "", headers or {})
  end)
  if not ok then
    SafeLog('[lua] RestApiPost failed: ' .. path .. ' err=' .. tostring(res))
    return nil
  end
  return res
end

local function SafeHttpPost(url, body, headers)
  local ok, res = pcall(function()
    return HttpPost(url, body or "", headers or {})
  end)
  if not ok then
    SafeLog('[lua] HttpPost failed: ' .. url .. ' err=' .. tostring(res))
    return nil
  end
  return res
end

-- 외부 HTTP 재시도 유틸
local function TryPostWithRetry(url, body, headers, retries, delay_ms)
  retries = retries or 3
  delay_ms = delay_ms or 2000  -- milliseconds
  for i = 1, retries do
    local resp = SafeHttpPost(url, body, headers)
    if resp then
      return resp
    end
    SafeLog('[lua] HttpPost retry ' .. i .. '/' .. retries .. ' to ' .. url)
    Sleep(delay_ms)
  end
  return nil
end

-- ===== 상태 머신 유틸 =====
local AI_STATE_KEY = "ai_state"  -- queued | processing | done | error

local function GetAIState(studyId)
  local v = SafeRestApiGet('/studies/' .. studyId .. '/metadata/' .. AI_STATE_KEY)
  if v == nil or v == "" then return nil end
  if v == '"queued"' then return "queued" end
  if v == '"processing"' then return "processing" end
  if v == '"done"' then return "done" end
  if v == '"error"' then return "error" end
  return v
end

local function SetAIState(studyId, state)
  SafeRestApiPost(
    '/studies/' .. studyId .. '/metadata/' .. AI_STATE_KEY,
    '"' .. state .. '"',
    { ["Content-Type"] = "application/json" }
  )
end
-- =========================

-- AI 결과 Study 필터 개선
local function IsAIResultStudy(studyId)
  SafeLog('[lua] Checking if study is AI result: ' .. studyId)
  
  local seriesListJson = SafeRestApiGet('/studies/' .. studyId .. '/series')
  if not seriesListJson then
    SafeLog('[lua] Failed to get series list for study: ' .. studyId)
    return true  -- 보수적으로 스킵
  end
  local seriesList = ParseJson(seriesListJson)
  if not seriesList or #seriesList == 0 then
    SafeLog('[lua] No series found in study: ' .. studyId)
    return false
  end

  local totalSeries = #seriesList
  local otModalities = 0
  
  SafeLog('[lua] Checking ' .. totalSeries .. ' series for AI result markers')

  for i = 1, totalSeries do
    local seriesId = seriesList[i]
    if type(seriesId) ~= "string" then
      SafeLog('[lua] SeriesId not string, skip'); goto continue
    end

    local seriesInfoJson = SafeRestApiGet('/series/' .. seriesId)
    if not seriesInfoJson then goto continue end

    local seriesInfo = ParseJson(seriesInfoJson)
    if not seriesInfo or not seriesInfo["MainDicomTags"] then goto continue end

    local modality = seriesInfo["MainDicomTags"]["Modality"]
    local seriesDescription = seriesInfo["MainDicomTags"]["SeriesDescription"] or ""

    -- 로깅 강화
    SafeLog('[lua] Series ' .. i .. ': modality=' .. (modality or "nil") .. 
            ', description="' .. (seriesDescription or "") .. '"')

    if modality == "OT" then 
      otModalities = otModalities + 1 
      SafeLog('[lua] ✓ OT modality found in series: ' .. seriesId)
    end

    if seriesDescription and (
       string.find(seriesDescription, "AI Analysis Result - Heatmap")) then
      SafeLog('[lua] AI keyword in SeriesDescription: ' .. seriesDescription)
      return true
    end

    ::continue::
  end

  -- OT 모달리티 비율에 따른 결정
  if otModalities > 0 and (otModalities / totalSeries) >= 0.5 then
    SafeLog('[lua] ✓ Majority OT series (' .. otModalities .. '/' .. totalSeries .. ')')
    return true
  end

  SafeLog('[lua] ✗ Not an AI result study')
  return false
end

-- (레거시) processed 플래그 체크는 남겨도 무방
local function IsMarkedAsProcessed(studyId)
  local ok, metadataJson = pcall(function()
    return RestApiGet('/studies/' .. studyId .. '/metadata/processed')
  end)
  if not ok then return false end
  return (metadataJson == "true" or metadataJson == '"true"')
end

function OnStableStudy(studyId)
  SafeLog('[lua] Study stable: ' .. studyId)

  -- reconstruct는 타임아웃 잦을 수 있으니 필요 시 FastAPI/후단에서 처리 권장
  -- pcall(function()
  --   HttpPost('http://localhost:8042/studies/' .. studyId .. '/reconstruct', '', {}, 5)
  -- end)

  -- 1) 상태 가드: queued/processing/done이면 재전송 금지
  local cur = GetAIState(studyId)
  if cur == "queued" or cur == "processing" or cur == "done" then
    SafeLog('[lua] skip by ai_state=' .. tostring(cur))
    return
  end
  -- error 상태면 재시도 허용(정책에 따라 스킵으로 바꿀 수 있음)

  -- 2) StudyInstanceUID 추출
  local tagsJson = SafeRestApiGet('/studies/' .. studyId .. '/shared-tags')
  if not tagsJson then SafeLog('[lua] no shared-tags, skip'); return end
  local tags = ParseJson(tagsJson)
  local studyInstanceUIDTag = tags["0020,000d"]
  if studyInstanceUIDTag == nil then
    SafeLog('[lua] no StudyInstanceUID, skip'); return
  end
  local studyInstanceUID = (type(studyInstanceUIDTag) == "table") and studyInstanceUIDTag["Value"] or studyInstanceUIDTag
  if not studyInstanceUID then SafeLog('[lua] invalid StudyInstanceUID, skip'); return end

  -- 3) AI 결과 Study면 스킵
  if IsAIResultStudy(studyId) then
    SafeLog('[lua] AI result study, skip: ' .. studyId); return
  end

  -- 4) (선택) 레거시 processed 플래그도 가드
  if IsMarkedAsProcessed(studyId) then
    SafeLog('[lua] legacy processed=true, skip'); return
  end

  -- 5) 원자적 가드: 먼저 ai_state=queued 기록
  SetAIState(studyId, "queued")

  local payload = {
    StudyInstanceUID = studyInstanceUID,
    Status = "stored",
    OrthancStudyID = studyId
  }
  local payload_json = DumpJson(payload)
  SafeLog('[lua] Sending webhook JSON: ' .. payload_json)

  -- 재시도 포함 전송
  local success, result = pcall(function()
    HttpPost("http://cxr-orchestrator:8000/api/webhook/on-store", payload_json, {
        ["Content-Type"] = "application/json"
    })
  end)
  if not success then
    SetAIState(studyId, "error")
    SafeLog('[lua] webhook post failed after retries; ai_state=error')
    return
  end

  SafeLog('[lua] webhook accepted: ' .. tostring(result))
  -- 이후 processing/done/error 전이는 FastAPI가 Orthanc 메타데이터에 기록
end

function OnStoredInstance(instanceId, tags, metadata)
  SafeLog('[lua] Instance stored: ' .. instanceId)
end