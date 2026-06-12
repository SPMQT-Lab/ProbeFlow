# ROI Manual Workflow Checklist

Manual regression checklist for the image viewer's ROI tools. Run it on a
representative STM image after changes to the canvas, ROI items, or the ROI
Manager — it covers the interactions that automated GUI tests exercise least
(drag, hover, half-finished drawing, tool switching).

## Rectangle ROI

- Draw a rectangle ROI.
- Select it from the canvas and from the ROI Manager.
- Move it in pan mode by dragging the active ROI.
- Rename it from the ROI Manager and from the canvas context menu.
- Delete it with the Delete key and from the context menu.
- Run FFT on the rectangle ROI and confirm it uses the selected region.
- Run histogram and background-subtraction region workflows where available and confirm they use the selected ROI.

## Line ROI

- Draw a line ROI and confirm the line-profile panel appears.
- Confirm the panel title identifies the active line ROI by name/id.
- Select a different line ROI and confirm the profile updates.
- Select a non-line ROI while the line tool is not active and confirm the profile panel clears or hides.
- Move the active line ROI and confirm the profile refreshes.
- Delete the active line ROI and confirm the profile panel clears or hides.

## Polygon ROI

- Click vertices and press Enter with at least three vertices; confirm the polygon closes.
- Double-click with at least three vertices; confirm the polygon closes without a duplicate final vertex.
- Press Escape while drawing and confirm preview items are removed.
- Press Enter or double-click with too few vertices and confirm no broken partial ROI remains.
- Change tools while a polygon is incomplete and confirm drawing cancels cleanly.

## Freehand ROI

- Drag to draw a freehand ROI and confirm it appears in the ROI Manager.
- Confirm its bounds/mask are sensible for the drawn path.
- Select, rename, move, and delete it.
- Press Escape or change tools while drawing and confirm the preview clears cleanly.

## Point ROI

- Click to place a point ROI.
- Confirm it appears in the ROI Manager.
- Select and delete it from the manager, canvas context menu, and Delete key.

## Tool Behaviour

- Keyboard shortcuts switch tools: R rectangle, E ellipse, G polygon, F freehand, L line, P point, 1-9 ROI selection, I invert, Delete, Escape.
- Escape cancels active drawing before closing the dialog/window.
- Pan mode left-drag on empty image pans the view.
- Pan mode drag on the active ROI moves the ROI.
- Middle mouse always pans.
- ROI drag-move does not conflict with panning.
- Hover highlight is visible and does not interfere with drag-moving.
- Moving an ROI emits one persisted move on release, with no duplicate ROI or stale preview.

## Persistence

- Draw several ROI kinds, switch image/channel if supported, and return; confirm the ROI set is still correct.
- Close and reopen the folder when sidecar support is available; confirm ROIs return from the `.rois.json` sidecar.

## Context Menus

- Right-click an ROI on the canvas and in the ROI Manager.
- Confirm rename/delete/invert behave consistently.
- Confirm area-only operations are disabled for line and point ROIs.
- Confirm line profile is enabled only for line ROIs.
- Confirm FFT, histogram, line-profile, and background entries either work on the selected ROI or are disabled/absent when not implemented.
