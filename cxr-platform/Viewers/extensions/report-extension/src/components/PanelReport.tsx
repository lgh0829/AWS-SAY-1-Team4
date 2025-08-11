import React, { useState } from 'react';

export default function PanelReport({ servicesManager }) {
  const [reportText, setReportText] = useState('');
  const [patientInfo, setPatientInfo] = useState({
    name: 'John Doe',
    id: 'PT-12345',
    age: '45',
    gender: 'M'
  });

  const handleReportChange = (e) => {
    setReportText(e.target.value);
  };

  const handleSave = () => {
    console.log('Report saved:', reportText);
    alert('Report has been saved!');
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: '#000',
      color: '#fff',
      overflow: 'hidden'
    }}>
      {/* Header */}
      <div style={{
        padding: '0.75rem 1rem',
        borderBottom: '1px solid #333',
        background: '#111'
      }}>
        <h3 style={{
          fontSize: '1rem',
          fontWeight: 'bold',
          margin: 0
        }}>Report</h3>
      </div>

      {/* Patient Info */}
      <div style={{
        padding: '0.75rem 1rem',
        borderBottom: '1px solid #333',
        fontSize: '0.875rem'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
          <span><strong>Patient:</strong> {patientInfo.name}</span>
          <span><strong>ID:</strong> {patientInfo.id}</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span><strong>Age:</strong> {patientInfo.age}</span>
          <span><strong>Gender:</strong> {patientInfo.gender}</span>
        </div>
      </div>

      {/* Report Content */}
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        padding: '1rem',
        overflow: 'hidden'
      }}>
        <textarea
          value={reportText}
          onChange={handleReportChange}
          placeholder="Enter report details here..."
          style={{
            flex: 1,
            width: '100%',
            background: '#242424',
            color: '#fff',
            border: '1px solid #444',
            borderRadius: '4px',
            padding: '0.75rem',
            resize: 'none',
            fontFamily: 'monospace',
            fontSize: '0.9rem',
            minHeight: '150px'
          }}
        />

        {/* Actions */}
        <div style={{
          marginTop: '1rem',
          display: 'flex',
          justifyContent: 'space-between'
        }}>
          <button
            onClick={() => setReportText('')}
            style={{
              padding: '0.5rem 1rem',
              background: '#555',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Clear
          </button>

          <button
            onClick={handleSave}
            style={{
              padding: '0.5rem 1.5rem',
              background: '#3498db',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold'
            }}
          >
            Save Report
          </button>
        </div>
      </div>
    </div>
  );
}
