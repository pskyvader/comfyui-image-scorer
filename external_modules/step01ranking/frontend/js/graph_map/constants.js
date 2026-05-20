function showError(msg) {
    console.error(msg);
    let el = document.getElementById("app-toast");
    if (!el) {
        el = document.createElement("div");
        el.id = "app-toast";
        el.style.cssText = "position:fixed;bottom:20px;right:20px;background:#ef4444;color:#fff;padding:12px 20px;border-radius:8px;z-index:99999;font-size:13px;font-family:monospace;max-width:500px;white-space:pre-wrap;display:none;";
        document.body.appendChild(el);
    }
    el.textContent = msg;
    el.style.display = "block";
    clearTimeout(el._hide);
    el._hide = setTimeout(() => el.style.display = "none", 6000);
}

// =============================================
// PHYSICS — force simulation parameters
// Used in: simulation.js
// =============================================
const PHYSICS = {
    node: {
        chargeStrength: -50, // D3 many-body repulsion (simulation.js:98,231)
        chargeDistanceMax: 100, // repulsion cutoff in world units (simulation.js:139)
        gravityStrength: 30, // score-based vertical pull magnitude (simulation.js:140)
        scoreExponent: 2, // pow() on normalized score bias (simulation.js:508)
        neutralDeadZone: 0, // score range around 0.5 with no vertical force (simulation.js:503)
        collisionRadius: 8, // base collision radius in world units (simulation.js:311)
        isolatedMultiplier: 0.85, // charge multiplier for degree-0 nodes (simulation.js:91,224)
        centerSpread: 20, // random spread around world center on spawn (simulation.js:94-95,227-228)
        batchAlpha: 0.9, // simulation alpha when adding a new node batch (simulation.js:62)
    },
    link: {
        linkDistance: 1, // base ideal link length in world units (simulation.js:238)
        linkStrength: 1, // base link stiffness 0-1 (simulation.js:271)
        linkStrengthFinalFactor: 1, // final link stiffness factor (simulation.js:38,291)
        scoreDistanceMultiplier: 10000, // score gap weight in link distance calc (simulation.js:239)
        countDistanceMultiplier: 10, // comparison count weight in link distance calc (simulation.js:240)
        maxLinkDistance: 1000000, // world units ceiling (simulation.js:147,212)
        linkConstraintPull: 0.5, // how hard to pull back links past tolerance (simulation.js:156,379)
        stretchTolerance: 1.08, // link length multiplier before constraint kicks in (simulation.js:396)
    },
    world: {
        velocityDecay: 0.01, // D3 velocity friction per tick (simulation.js:141)
        alphaDecay: 0.001, // D3 cooling rate (simulation.js:142)
        useBouncyBounds: true, // bounce off edges instead of clamp (simulation.js:439)
        boundaryBounce: 5, // restitution when bouncing off bounds (simulation.js:445)
        minWorldSize: 1000, // minimum world dimension (simulation.js:34-35,210)
        maxWorldSize: 1200000, // maximum world dimension (simulation.js:212)
        worldScale: 200, // world size = ceil(sqrt(n)) * this (simulation.js:148)
        boundaryPadding: 0, // margin inside world edges (simulation.js:149)
        width: 800, // viewport width fallback (simulation.js:210)
        height: 600, // viewport height fallback (simulation.js:210)
    },
};

// =============================================
// SCALE — multipliers applied to PHYSICS values
// Used in: simulation.js
// =============================================
const SCALE = {
    node: {
        chargeDistanceMultiplier: 1.0, // 100 * 1 = 100 unit radius (simulation.js:139)
        gravityMultiplier: 2.0, // stronger vertical pull (simulation.js:140)
    },
    link: {
        distanceMultiplier: 1.0, // link distance (simulation.js:143)
        strengthMultiplier: 0.5, // half link stiffness (simulation.js:144)
        scoreDistMultiplier: 1.0, // score gap weight (simulation.js:145)
        countDistMultiplier: 1.0, // count weight (simulation.js:146)
        maxDistanceMultiplier: 1.0, // max link distance (simulation.js:147)
        constraintPullMultiplier: 1.0, // link constraint pull (simulation.js:156)
    },
    world: {
        velocityDecayMultiplier: 1.0, // 3x friction (simulation.js:141)
        alphaDecayMultiplier: 1.0, // 3x cooling rate (simulation.js:142)
        worldScaleMultiplier: 1.0, // world scale (simulation.js:148)
        boundaryPaddingMultiplier: 1.0, // boundary padding (simulation.js:149)
    },
};

// =============================================
// PERFORMANCE — performance tuning & thresholds
// Used in: simulation.js, canvas_renderer.js
// =============================================
const PERFORMANCE = {
    link: {
        linkStrideDivisor: 45000, // link stride for large graphs (simulation.js:151)
        linkStrideLargeMultiplier: 4 / 3, // extra stride for large graphs (simulation.js:151)
        drawLinksMaxLinks: 1800000, // max links before link drawing disabled (simulation.js:133)
        perLinkOpacityMaxLinks: 100000, // max links before per-link opacity disabled (simulation.js:135)
    },
    constraint: {
        constraintPhasesDivisor: 90000, // constraint phase stride (simulation.js:154)
        constraintPhasesLargeMultiplier: 5 / 9, // extra phase stride for large graphs (simulation.js:154)
    },
    batch: {
        batchTickDivider: 1, // add next batch every N sim ticks (simulation.js:332)
        nodeBatchMin: 200, // min nodes per sim batch (simulation.js:31)
        nodeBatchDivisor: 100, // ceil(nodes/this) if larger than min (simulation.js:31)
        progressiveRevealMinNodes: 5000, // min nodes for progressive reveal (simulation.js:134)
    },
    render: {
        renderIntervalMs: 16, // throttle for onTick renders ~60fps (simulation.js:347)
        mediumRenderIntervalMsMultiplier: 2, // medium graph: 2x render interval (simulation.js:138)
        largeRenderIntervalMsMultiplier: 3, // large graph: 3x render interval (simulation.js:138)
    },
    graph: {
        mediumGraphMinWorkUnits: 9000, // medium graph threshold (simulation.js:109)
        largeGraphMinWorkUnits: 20000, // large graph threshold (simulation.js:110)
        linkWorkWeight: 1.0, // link weight in workUnits calc (simulation.js:108)
        startPausedWorkUnits: 9000, // start paused threshold (simulation.js:157)
    },
    node: {
        labelsMaxNodes: 900, // max nodes before labels disabled (simulation.js:129)
        tooltipsMaxNodes: 12000, // max nodes before tooltips disabled (simulation.js:132)
        pointerHitsMaxNodes: 50000, // max nodes before pointer hits disabled (simulation.js:123)
    },
};

// =============================================
// RENDER — visual appearance settings
// Used in: simulation.js, canvas_renderer.js, main.js, interactions.js
// =============================================
const RENDER = {
    node: {
        baseRadius: 8, // base node radius in world units (main.js:226)
        radiusMultiplier: 3, // radius += sqrt(count) * this (main.js:226)
        minScreenRadius: 2.5, // minimum px radius at any zoom (canvas_renderer.js:278,305,332,358)
        labelTruncateLength: 20, // max chars before "..." (main.js:229)
        colorDomain: [0, 0.45, 0.5, 0.54, 1], // score color stops (main.js:214)
        colorRange: ["#ef4444", "#facc15", "#facc15", "#facc15", "#22c55e"], // score colors red→yellow→green (main.js:215)
        defaultOpacity: 1, // default node opacity (main.js:237)
        borderStroke: "rgba(255, 255, 255, 0.8)", // node border color (canvas_renderer.js:329)
        borderLineWidth: 0.9, // node border width (canvas_renderer.js:330)
        selectedLineWidth: 2, // selected node border width (canvas_renderer.js:302)
        highlightColor: "#f472b6", // pink stroke for selected nodes (canvas_renderer.js:303)
    },
    link: {
        linkColor: "#ffffff", // link stroke color (canvas_renderer.js:213)
        chainColor: "#60a5fa", // chain link stroke color (canvas_renderer.js)
        linkLineWidth: 1, // link stroke width (canvas_renderer.js:212)
        chainLineWidth: 2, // chain link stroke width (canvas_renderer.js)
        linkViewportPadding: 10, // culling margin for link opacity pass (canvas_renderer.js:246)
        linkOpacityMinDist: 10, // distance where opacity starts decreasing (canvas_renderer.js:255)
        linkOpacityMaxDist: 100000, // distance where opacity hits minimum (canvas_renderer.js:254)
        linkOpacityMin: 0.1, // minimum opacity (canvas_renderer.js:255)
        linkOpacityMax: 1.0, // maximum opacity (canvas_renderer.js:216,255)
        linkVisibilityZoomThreshold: 0.01, // zoom below which links fade (simulation.js:137, canvas_renderer.js:202)
        smallLinkVisibilityZoomThresholdMultiplier: 0, // small graphs: links always visible (simulation.js:137)
        largeLinkVisibilityZoomThresholdMultiplier: 0.01, // large graphs: higher zoom threshold (simulation.js:137)
    },
    arrow: {
        arrowColor: "rgba(248, 250, 252, 0.62)", // arrow fill color (canvas_renderer.js:378)
        arrowViewportPadding: 12, // arrow culling margin (canvas_renderer.js:393)
        arrowMinLengthSq: 36, // skip arrows on short links (canvas_renderer.js:376)
        arrowheadSize: 2.8, // arrowhead size (canvas_renderer.js:406)
        arrowheadWingRatio: 0.7, // arrowhead wing spread (canvas_renderer.js:420,421)
        arrowMidOffsetRatio: 0.07, // arrow drawn fraction past midpoint (canvas_renderer.js:406)
        arrowCap: 2000, // max arrows per frame (interactions.js:41)
    },
    label: {
        labelColor: "rgba(226, 232, 240, 0.92)", // label text color (canvas_renderer.js:351)
        labelCap: 500, // max labels per frame (main.js:153, interactions.js:40)
        labelFontSizeMin: 6, // minimum font size (canvas_renderer.js:362)
        labelFontSizeMax: 16, // maximum font size (canvas_renderer.js:362)
        labelFontSizeMultiplier: 1.6, // fontSize = clamp(min, round(r*this), max) (canvas_renderer.js:362)
        labelOffset: 3, // label offset from node edge (canvas_renderer.js:367)
        labelViewportPadding: 20, // label culling margin (canvas_renderer.js:359)
    },
    border: {
        nodeBorderViewportPadding: 1, // border culling margin (canvas_renderer.js:336)
    },
    bounds: {
        strokeColor: "rgba(139, 92, 246, 0.8)", // world boundary stroke color (canvas_renderer.js:191)
        lineWidth: 2, // world boundary stroke width (canvas_renderer.js:192)
        dashArray: [12, 6], // world boundary dash pattern (canvas_renderer.js:193)
        padding: 150, // world boundary padding (canvas_renderer.js:78-79, main.js:303-307)
    },
    viewport: {
        hitTestPadding: 10, // makes clicking small nodes easier (canvas_renderer.js:130)
    },
};

// =============================================
// CAMERA — zoom/viewport behavior and thresholds
// Used in: canvas_renderer.js, controls.js, interactions.js
// =============================================
const CAMERA = {
    viewportPadding: 24, // viewport culling margin (canvas_renderer.js:182)
    labelThreshold: 0.25, // zoom level to show labels (interactions.js:36)
    arrowThreshold: 0.15, // zoom level to show arrows (interactions.js:37)
    borderThreshold: 0.2, // zoom level to show node borders (interactions.js:38)
    linkThreshold: 0.05, // zoom level to show links (interactions.js:39)
    focusScale: 1.5, // zoom scale when focusing a node (interactions.js:93)
    focusDuration: 750, // ms for focus animation (interactions.js:89)
    fitPadding: 0.8, // fraction of viewport used when fitting world (controls.js:238)
    maxFitScale: 4, // max zoom when fitting world (controls.js:238)
    fallbackScale: 0.05, // zoom when world has no dimensions (controls.js:233)
    minScale: 0.001, // minimum zoom scale (canvas_renderer.js:126,182,427)
    zoomExtent: [0.001, 5], // min/max zoom scale factor (interactions.js:21)
};

// =============================================
// CHAIN — chain layout and phased reveal settings
// Used in: simulation.js
// =============================================
const CHAIN = {
    anchorScoreThreshold: 0.02,
    maxSpreadFraction: 0.55,
    minSpread: 0.1,
    phaseDelay: 400,
    relaxDelay: 30,
    relaxIterations: 80,
    relaxStepSize: 0.008,
    relaxStepDecay: 0.96,
    topBottomSpread: 0.25,
};

// =============================================
// CONTROLS — user interaction settings
// Used in: simulation.js, canvas_renderer.js, main.js, controls.js, interactions.js
// =============================================
const CONTROLS = {
    pausedLinkedDragInfluence: 10, // linked nodes shift by this when dragging while paused (simulation.js:471-472)
    transitionDuration: 750, // ms for animated zoom/fit transitions (main.js:290, controls.js:229,241)
    enablePointerHits: false, // mouse hit-testing on nodes (simulation.js:123, canvas_renderer.js:112)
};
