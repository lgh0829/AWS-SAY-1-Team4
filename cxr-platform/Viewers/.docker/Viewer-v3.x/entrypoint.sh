#!/bin/sh

if [ -n "$APP_CONFIG" ]; then
  echo "Using custom APP_CONFIG environment variable"
  
  # 환경 변수로 지정된 위치에서 파일 확인
  if [ -f "$APP_CONFIG" ]; then
    # 볼륨으로 마운트된 파일이면 복사본 생성 후 처리
    if [ -f /usr/share/nginx/html/app-config.js ]; then
      # 파일이 읽기 전용인지 확인
      touch /usr/share/nginx/html/app-config.js 2>/dev/null
      if [ $? -ne 0 ]; then
        echo "app-config.js is read-only, creating a writable copy..."
        cp /usr/share/nginx/html/app-config.js /tmp/app-config.js
        cat /tmp/app-config.js > /usr/share/nginx/html/app-config.js.tmp
        mv /usr/share/nginx/html/app-config.js.tmp /usr/share/nginx/html/app-config.js 2>/dev/null || true
      fi
    fi
  fi
else
  echo "Not using custom APP_CONFIG"
fi

# 압축 처리
if [ -f /usr/share/nginx/html/app-config.js ]; then
  echo "Detected app-config.js. Ensuring .gz file is updated..."
  if [ -f /usr/share/nginx/html/app-config.js.gz ]; then
    rm -f /usr/share/nginx/html/app-config.js.gz 2>/dev/null || true
  fi
  gzip -c /usr/share/nginx/html/app-config.js > /usr/share/nginx/html/app-config.js.gz 2>/dev/null || true
  echo "Compressed app-config.js to app-config.js.gz"
else
  echo "No app-config.js file found. Skipping compression."
fi

echo "Starting Nginx to serve the OHIF Viewer on ${PUBLIC_URL}"
exec "$@"