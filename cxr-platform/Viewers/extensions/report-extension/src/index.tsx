import { id } from './id';
import getPanelModule from './getPanelModule';

/**
 * 리포트 기능을 위한 확장 프로그램
 */
export default {
  /**
   * 필수 속성. 모든 확장 프로그램에서 고유한 값이어야 합니다.
   */
  id: 'extension-report',

  /**
   * 리포트 패널을 추가하는 PanelModule
   */
  getPanelModule,
};
