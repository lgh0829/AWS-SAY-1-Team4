import React, { useState } from 'react';

export default function PanelReport({ servicesManager }) {
  const [reportText, setReportText] = useState('');

  const handleReportChange = (e) => {
    setReportText(e.target.value);
  };

  const handleSave = () => {
    console.log('Report saved:', reportText);
    alert('리포트가 저장되었습니다!');
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 bg-black">
        <h3 className="text-white text-lg font-bold mb-4">리포트</h3>
        <div className="flex flex-col gap-3">
          <textarea
            value={reportText}
            onChange={handleReportChange}
            placeholder="여기에 리포트를 입력하세요..."
            className="w-full h-64 bg-black border border-gray-600 rounded p-2 text-white"
            style={{ resize: 'vertical' }}
          />
          <button
            onClick={handleSave}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            리포트 저장
          </button>
        </div>
      </div>
    </div>
  );
}
