# Local model preview

Serve the parent `c151-delivery` directory over HTTP, then open `/preview/` in the browser. Do not open the HTML with a `file://` URL: browser module imports and Draco workers need HTTP.

From the delivery directory with Node.js installed:

```console
node scripts/serve.mjs
```

Open `http://127.0.0.1:43151/preview/`. Three.js and its Draco decoder load from the bundled `vendor/three` directory; the viewer has no remote fonts, scripts or API dependencies.

The viewer supports cab, intermediate and six-car configurations; door opening, wireframe, orbit, and four camera presets. Dimensions and triangle counts appear below the scene. Its ground, track and studio lights are preview helpers and are not included in either exported GLB.

## Inspection hooks

- `window.previewReady`: true after both GLBs have decoded and entered the scene.
- `window.previewLoadPromise`: resolves when loading completes.
- `window.previewState`: `mode`, `view`, `doors`, per-asset triangle/bounds/material statistics, current visible statistics, and any loading error.
- `window.setPreviewMode('cab' | 'intermediate' | 'consist')`
- `window.setPreviewView('perspective' | 'side' | 'front' | 'roof')`
- `window.setDoorsOpen(0…1)`
- `window.previewRenderer`, `window.previewScene`, `window.previewCamera`, and `window.previewControls` for local inspection.

After changing a control, allow one animation frame before capturing the canvas. The renderer preserves its drawing buffer for screenshots.
