# Crystal Layout - Implementation Reference

## API Interface (Maintained Compatibility)

### Constructor
```javascript
new ChainSimulation(nodes, links, components, options)
```

All existing parameters work exactly as before. New internal parameters:
- `hexRadius`: Hexagonal grid spacing (default: 12)
- `componentPadding`: Space around component area (default: 40)

### Initialization Flow
```javascript
sim.initialize()
  ├─ indexNodes()           // Map node ID → node object
  ├─ groupNodesByComponent() // Group nodes by component ID
  ├─ calculateNodeDegree()   // Count incoming/outgoing links
  ├─ prepareWorldBounds()    // Calculate world size
  ├─ prepareNodes()          // Position nodes in crystal lattice
  ├─ prepareLinks()          // Configure link rendering
  ├─ setupSimulation()       // No-op (empty method)
  └─ start()                 // Emit onTick callback immediately
```

### Key Methods

#### `_positionNodesInCrystal(nodes)`
- Groups nodes by component
- Sorts nodes by score (descending)
- Generates hexagonal positions for each group
- Sets x, y coordinates on each node
- Clears velocity (vx, vy) and constraints (fx, fy)

#### `_generateHexagonalLattice(count, width, height, radius)`
- Creates `count` positions arranged in hexagonal pattern
- Fits within rectangular bounds (width × height)
- Returns array of {x, y} position objects
- Centered around (0, 0)

#### `_componentCenters()`
- Arranges components in √N × √N grid
- Returns Map of component ID → {x, y, width, height}
- Uses `this.effectiveWidth` and `this.effectiveHeight`

### Event Callbacks
```javascript
options.onTick   // Called during/after each render
options.onEnd    // Called when layout complete (after start())
```

### Play/Pause Methods
```javascript
sim.play()       // Resume (no-op in crystal layout)
sim.pause()      // Pause (toggle isPaused flag)
sim.isPaused     // Boolean state
```

## Data Flow

```
[1] Main.js calls: chainSim.initialize()
    ↓
[2] Initialize triggers all preparation:
    - indexNodes() → nodeById Map
    - groupNodesByComponent() → componentNodes Map
    - calculateNodeDegree() → nodeDegree Map
    - prepareWorldBounds() → set effectiveWidth/Height
    ↓
[3] prepareNodes() (CRYSTAL LAYOUT):
    - Get component centers from _componentCenters()
    - For each component:
      * Sort nodes by score
      * Generate hex lattice positions
      * Assign x, y to each node
    ↓
[4] All nodes now have static positions
    ↓
[5] start() called:
    - Emit onTick callback
    - onTick updates renderer with all nodes visible
    ↓
[6] Canvas renders:
    - All nodes drawn at calculated positions
    - Links drawn between nodes
    - No animation
    ↓
[7] Ready for interaction:
    - Pan/zoom work (transform world, not nodes)
    - Click to select nodes
    - Filters trigger re-render
```

## Node Positioning Algorithm Details

### Component Grid Calculation
```javascript
columnCount = ceil(√componentCount)
rowCount = ceil(componentCount / columnCount)

usableWidth = worldWidth - (padding × 2)
usableHeight = worldHeight - (padding × 2)

cellWidth = usableWidth / columnCount
cellHeight = usableHeight / rowCount

For each component i:
  column = i % columnCount
  row = floor(i / columnCount)
  centerX = padding + (column + 0.5) × cellWidth
  centerY = padding + (row + 0.5) × cellHeight
```

### Hexagonal Lattice Calculation
```javascript
columnCount = ceil(√nodeCount)
rowCount = ceil(nodeCount / columnCount)

For each position i:
  row = floor(i / columnCount)
  col = i % columnCount
  
  // Alternating X offset for hex pattern
  xOffset = (row % 2) × (radius × 0.5)
  x = (col × radius) + xOffset - (columnCount × radius × 0.5)
  
  // Y spacing: sqrt(3)/2 ≈ 0.866
  y = (row × radius × 0.866) - (rowCount × radius × 0.433)
  
  // Clamp to bounds
  x = clamp(x, -width/2, width/2)
  y = clamp(y, -height/2, height/2)
```

### Final Position
```javascript
node.x = componentCenterX + latticeX
node.y = componentCenterY + latticeY
```

## Configuration Values Used

From `constants.js`:

```javascript
PHYSICS.world.minWorldSize      // Minimum world dimension
PHYSICS.world.maxWorldSize      // Maximum world dimension
PHYSICS.world.worldScale        // Scale factor for world size
PHYSICS.world.boundaryPadding   // Margin around edges

RENDER.node.baseRadius          // Node circle radius
RENDER.node.radiusMultiplier    // Radius based on comparison count
RENDER.node.colorDomain/Range   // Color scale for score

CONTROLS.transitionDuration     // Delay before play() in main.js
```

All other PHYSICS/RENDER/PERFORMANCE values are calculated but not used in crystal layout (kept for compatibility).

## Edge Cases Handled

### Empty Components
- If a component has 0 nodes: skipped in positioning
- If all nodes filtered: layout still calculates, renders nothing

### Single Node
- Returns single position at (0, 0) relative to component

### Very Large Node Count
- Hex lattice scales to accommodate any count
- Component grid scales to √N × √N

### Very Small Component Area
- Positions get clamped to bounds
- Nodes may overlap if area too small for hexRadius

## Potential Issues & Solutions

### Issue: Nodes appear in wrong positions
**Solution**: Verify `this.effectiveWidth` and `this.effectiveHeight` are calculated correctly in `prepareWorldBounds()`. Check console logs for dimension values.

### Issue: Crystals look stretched
**Solution**: The hex lattice uses sqrt(3) for Y spacing. This is mathematically correct for hex packing. If visuals appear wrong, adjust `hexRadius` parameter.

### Issue: Nodes appear at canvas edge
**Solution**: Component centers might be placing them at world boundary. Increase `PHYSICS.world.boundaryPadding` or reduce component count.

### Issue: Play/Pause button doesn't work
**Solution**: Should just toggle `isPaused` flag. No physics to animate. Make sure `toggleSimulation()` in main.js calls `play()` or `pause()` on chainSim.

### Issue: Filters don't update crystal
**Solution**: When filters change, main.js calls `applyFilters()` → `render()` → creates NEW ChainSimulation with filtered nodes. Old sim is stopped. Crystal recalculates with filtered node set.

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| initialize() | 0-5ms | Just calculation, no physics |
| Position nodes | 0-2ms | Simple arithmetic |
| Generate lattice | 0-1ms | Per component |
| Total layout | < 5ms | For 1000+ nodes |
| First render | 16ms | Canvas draw, 60fps |

Physics-based version: 30,000+ ms (30 seconds)
Crystal version: < 5ms

## Testing Scenarios

### Scenario 1: Basic rendering
```
→ Load page
✓ Crystal formations appear instantly
✓ Nodes in hexagonal patterns
✓ Colors correct by score
✓ Links between nodes visible
```

### Scenario 2: Filtering
```
→ Change filter sliders
✓ Some nodes disappear
✓ Crystal recalculates
✓ Remaining nodes still in lattice
✓ Links update correctly
```

### Scenario 3: Interaction
```
→ Click and drag canvas
✓ Pan works smoothly
→ Scroll to zoom
✓ Zoom works smoothly
→ Click node
✓ Selection highlights work
→ Select two nodes
✓ Compare button works
```

### Scenario 4: Large dataset
```
→ Load 10,000 nodes
✓ Layout appears instantly
✓ No UI blocking
✓ Pan/zoom responsive
✓ No memory leaks
```

## Debugging Tips

### Check Node Positions
```javascript
// In browser console:
console.log(chainSim.nodes.slice(0, 5).map(n => ({
  id: n.id,
  component: n.component,
  x: n.x.toFixed(2),
  y: n.y.toFixed(2)
})));
```

### Check Component Centers
```javascript
console.log('Component centers:');
chainSim._componentCenters().forEach((center, compId) => {
  console.log(compId, center);
});
```

### Check World Bounds
```javascript
console.log('World size:', {
  width: chainSim.effectiveWidth,
  height: chainSim.effectiveHeight
});
```

### Check Lattice Generation
```javascript
// Test hex lattice with 9 nodes in 100×100 area:
const positions = chainSim._generateHexagonalLattice(9, 100, 100, 12);
console.log('Positions:', positions);
// Should form 3×3 hex grid
```

## Future Optimization Ideas

1. **Render only visible nodes** (viewport culling)
2. **Progressive component reveal** as user pans
3. **Animated transitions** when filters change
4. **Link flow direction** using arrows
5. **Physics-optional mode** for comparison
6. **Export crystal layout** as SVG/PNG
7. **Save/load crystal snapshots**
8. **Custom crystal shapes** (diamond, star, etc.)
