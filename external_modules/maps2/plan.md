# Graph Map — Fresh Start Plan

## What Exists (Current `maps/`)

### Data
- ~30k image nodes with similarity scores, grouped into components and chains
- ~377k directed edges between comparisons (winner→loser)

### Rendering
- **Three.js** (WebGL) for GPU-accelerated rendering

### Controls 
- Pan: left-drag via OrbitControls (WebGL) or custom snapshot-drag (2D fallback)
- Zoom: scroll wheel + zoom-in/zoom-out buttons
- Reset view button (resets for full visibility of the map)
- Toggle labels, toggle main chains, toggle regular links
- Pause/play buttons
- Filter sliders: component size, chain depth, node comparisons
- Tooltip on hover, details panel on click
- HUD: zoom scale + view coordinates

external_modules/server/frontend/html/index.html   # SPA shell with CDN script tags
external_modules/server/frontend/js/index.js        # SPA router (route "chains" → maps.html)

---

## What I Want

### Core Requirements
- **GPU acceleration** — Three.js WebGL as primary renderer; if WebGL unavailable, show an error and don't do anything else.
- **Interactable** — left-drag pan, scroll zoom, click for details, hover for tooltips
- **Responsive** — page must not freeze during initial load; 
- **Reliable resize** — resize never breaks; canvas adapts, content re-renders 
- on click on node show node details: image, comparisons, score, add to comparison button. highlight selected node
- on click on chain, show chain details: how many main chains it has, chain id. it should highlight chain links and nodes, with extra highlight to the nodes that have that chain as its own main.

### Layout
- nodes grouped by components
- Node positions by score (Y axis) and links (x and y axis).
- basic links are not the same as main chain links. basic are more opaque and don't contribute in any way to node positioning. main chain links are part of a full main chain, spanning from the very best nodes in a component to the very bottom, and contribute to give shape to the different components. so and ideal component would have a diamont structure (or maybe curved but still close to diamond), in which its main chains span from top to bottom, and the links belonging to those chains have fixed sizes based onnthe score difference.
- it must look like a network, or crystalline structure, or diamond.

- Slider changes recompute layout immediately and re-render
- background grid every 10 px, fades out when zooming out, then bigger grid every 100px that also fades out, then 1000px,10k,100k, etc up to full map size. so grids should only overlap with the next/previous grid size, and a max of 2 grids should appear at the same time
- borders of the map.
- map size based on nunber of nodes. adjustable in front end via slider.



### Visual
- Links behind nodes (nodes visible on top)
- Selected nodes highlighted (yellow/amber)
- Main chains visible as brighter/thicker lines
- Regular links as thin, semi-transparent lines
- Tooltip on hover shows node ID
- Details panel on click shows node info

### Architecture (Clean)
```
ThreeGraphRenderer
  ├── WebGL path (THREE.WebGLRenderer + OrbitControls)
  │   ├── PointsGeometry for nodes (29K points, instanced)
  │   ├── LineSegments for links
  │   ├── Main chains as a second LineSegments (different color/width)
  │   └── Raycaster for hit-testing


Rendering rules:
  - render only on controls.change (no rAF loop)
  - resize → resize canvas → (WebGL: render;)
  
Event flow:
  - WebGL: OrbitControls handles drag/zoom natively
           canvas.addEventListener('mousemove') for hover
           canvas.addEventListener('click') for details
```

### Controls (HTML buttons, always visible)
- Zoom in / Zoom out
- Reset view
- Toggle labels
- Toggle main chains
- Toggle regular links
- Refresh (re-fetch data)
- Filters accordion: component size, chain depth, node comparisons, collapsible, node type
- Layout accordion: component spacing, score spread, jitter, global spread
- Stats: node count, link count, component count, chain count
- Legend: color meaning, main chain vs regular link
- HUD: zoom level, view center coordinates


details (all these should be sliders to control in front end):
node base radius 5px
node actual size: base radius * links count^power selector

map area per node: 200x200px > map full height and width= sqrt(area*node count)
links length: score difference (from 0 to 1) * map height * node count in component * multiplier

filters (to show or hide any given component):
nodes count min/max
chain length min/max
comparison count min/max

all of these filters should only define if an entire component should be shown or hidden.




options:
- render using physics. need to define more sliders like strenght/elasticity of links, buoyancy of the scores, repeling force of the nodes (force and reach). then just start the nodes in random positions and let the physics take charge.

- render precalculating fixed positions. wouldnt need to add more sliders, but it needs to set positions across the map: for each component assing a part of the map (vertical and horizontal, since i want allcomponents distributed across the wntire map in both directions) with the size based on their nodes count. inside each: calculate shortest and longest chains, position the shortest in the middle and then the logers more and more outside. the top nodes should be in the top edge of that sub map, lowest bottom edge, and scores of exactly 0.5 across the exact horizontal middle of the submap.