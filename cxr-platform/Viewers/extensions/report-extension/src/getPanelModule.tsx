import PanelReport from './components/PanelReport';

export default function getPanelModule() {
  return [
    {
      name: 'report',
      iconName: 'clipboard',
      iconLabel: 'Report',
      label: 'Report',
      component: PanelReport,
    },
  ];
}
