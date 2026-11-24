import { describe, expect, it } from 'vitest';
import {
  buildNodePopupElement,
  buildTrackPopupElement,
} from '../src/components/SpatialMap';

describe('SpatialMap popup XSS safety', () => {
  it('renders telemetry strings strictly via textContent without HTML parsing', () => {
    const maliciousId = '<img src=x onerror=alert(1)>';
    const nodePopup = buildNodePopupElement(
      'Peer Mesh Node',
      maliciousId,
      50.45,
      30.52,
      '861e614bfffffff',
      15
    );

    expect(nodePopup.querySelector('img')).toBeNull();
    expect(nodePopup.textContent).toContain(maliciousId);

    const trackPopup = buildTrackPopupElement({
      trackId: '<script>alert(1)</script>',
      class: 'Shahed_Piston_UAV',
      lat: 50.451,
      lng: 30.521,
      vx: 35,
      vy: 15,
      speedMps: 38.1,
      headingDeg: 67,
      confidence: 0.92,
      snrDb: 19.5,
      nodeCount: 3,
      lastUpdated: Date.now(),
      history: [],
    });

    expect(trackPopup.querySelector('script')).toBeNull();
    expect(trackPopup.textContent).toContain('<script>alert(1)</script>');
  });
});
