# Crystal Layout - Visual Guide

## How Nodes Are Positioned

### Step 1: Component Grid Layout

```
┌─────────────┬─────────────┬─────────────┐
│             │             │             │
│  Component 0│  Component 1│  Component 2│
│             │             │             │
├─────────────┼─────────────┼─────────────┤
│             │             │             │
│  Component 3│  Component 4│  Component 5│
│             │             │             │
├─────────────┼─────────────┼─────────────┤
│             │             │             │
│  Component 6│  Component 7│  Component 8│
│             │             │             │
└─────────────┴─────────────┴─────────────┘
```

Components arranged in √N × √N grid (e.g., 9 components = 3×3)

### Step 2: Hexagonal Lattice Within Component

For a component with 9 nodes:

```
Sorted by score (★ = highest):

★ ★ ★         (score tier 1)
★ ★ ★         (score tier 2)
★ ★ ★         (score tier 3)

Arranged in hexagonal pattern:

    ●     ●     ●
  ●     ●     ●
    ●     ●     ●

Perfect hexagonal packing creates crystal structure
```

### Step 3: Final Layout

```
Full visualization with 3 components × 3 nodes each:

Component 0          Component 1          Component 2
  ●   ●               ●   ●               ●   ●
    ●                   ●                   ●
  ●   ●               ●   ●               ●   ●
    ●                   ●                   ●

Component 3          Component 4          Component 5
  ●   ●               ●   ●               ●   ●
    ●                   ●                   ●
  ●   ●               ●   ●               ●   ●
    ●                   ●                   ●

Component 6          Component 7          Component 8
  ●   ●               ●   ●               ●   ●
    ●                   ●                   ●
  ●   ●               ●   ●               ●   ●
    ●                   ●                   ●
```

## Node Color Coding (Unchanged)

- 🔴 Red: Low score (< 0.45)
- 🟡 Yellow: Medium score (0.45 - 0.54)
- 🟢 Green: High score (> 0.54)

## Interaction Flow

```
Load Data
    ↓
Apply Filters
    ↓
Render → ChainSimulation.initialize()
    ↓
Calculate component centers (grid layout)
    ↓
Position nodes in hexagonal lattice (per component)
    ↓
Emit onTick callback
    ↓
Canvas renders nodes + links
    ↓
Ready for interaction (pan/zoom/click)
```

## Customization Points

### In simulation.js constructor:

```javascript
// Change crystal density
this.hexRadius = 12;  // default: 12 units between nodes
// Smaller = denser, Larger = sparser

// Change component spacing
this.componentPadding = 40;  // default: 40 units padding
// Smaller = components closer, Larger = more space
```

### In prepareWorldBounds():

```javascript
// Change overall world size scaling
const scaledWorld = Math.ceil(Math.sqrt(Math.max(1, this.nodes.length)) * this.runtime.worldScale);
// this.runtime.worldScale is from SCALE configuration
```

## Performance Characteristics

| Aspect | Before (Physics) | After (Crystal) |
|--------|------------------|-----------------|
| Layout Time | ~30 seconds | Instant |
| CPU Usage | Continuous | Only on render |
| Memory | Growing vectors | Static positions |
| Consistency | Variable | Always same |
| Node Count | 0 to N (progressive) | All visible immediately |

## Example: 100 Nodes, 4 Components

```
Component arrangement: 2×2 grid
Nodes per component: 25 each

Component 0          Component 1
(25 nodes in          (25 nodes in
 5×5 hex grid)        5×5 hex grid)


Component 2          Component 3
(25 nodes in          (25 nodes in
 5×5 hex grid)        5×5 hex grid)

Each 5×5 grid creates perfect crystal formation
Total render time: < 100ms
```

## Debugging: Crystal Doesn't Look Right?

1. **Nodes overlapping?**
   - Increase `hexRadius` (try: 15)

2. **Too much white space?**
   - Decrease `hexRadius` (try: 8)

3. **Components too close?**
   - Increase `componentPadding` (try: 60)

4. **All nodes centered in one spot?**
   - Check `_componentCenters()` is calculating grid properly
   - Verify `this.effectiveWidth` and `this.effectiveHeight` are set

5. **Nodes outside visible area?**
   - Check component center calculations
   - Verify `node.x` and `node.y` are within world bounds
