const MAP_VISUALS = {
    nodeMinRadius: 3,
    highlightColor: "#f472b6",
    linkVisibilityZoomThreshold: 0.01,
    linkGlobalOpacity: 1,
    drawLinks: true,
    progressiveReveal: true,
    enablePointerHits: true,
};

const MAP_INTERACTION = {
    zoomExtent: [0.005, 5],
    transitionDuration: 750,
};

const MAP_FILTERS = {
    sliderMax: 20,
};

const SIMULATION_LINKS = {
    linkDistance: 10,
    linkStrength: 1,
    linkStrengthInitialFactor: 0.05,
    linkStrengthFinalFactor: 0.5,
    linkStrengthRampMs: 10000,
    linkStrengthRampCurve: 1.0,
    scoreDistanceMultiplier: 10000,
    countDistanceMultiplier: 10,
    maxLinkDistance: 1000000,
    linkConstraintSlack: 1.2,
    linkConstraintPull: 0.5,
    linkConstraintInitialSlackExtra: 1.9,
    scaling: {
        small: {},
        medium: {
            // linkDistance: 0.82,
            // linkStrength: 0.9,
            // scoreDistanceMultiplier: 0.75,
            // countDistanceMultiplier: 0.45,
            // maxLinkDistance: 0.5,
            // linkConstraintSlack: 0.92,
            // linkConstraintPull: 1.2,
            // linkStrengthRampMs: 0.8,
        },
        large: {
            // linkDistance: 0.68,
            // linkStrength: 0.78,
            // scoreDistanceMultiplier: 0.45,
            // countDistanceMultiplier: 0.2,
            // maxLinkDistance: 0.1,
            // linkConstraintSlack: 0.9,
            // linkConstraintPull: 1.5,
            // linkStrengthRampMs: 0.55,
        },
    },
};

const SIMULATION_NODES = {
    chargeStrength: -5,
    chargeDistanceMax: 1000,
    gravityStrength: 30,
    scoreExponent: 2,
    neutralDeadZone: 0,
    collisionRadius: 8,
    nodeJitter: 0,
    pausedLinkedDragInfluence: 10,
    scaling: {
        small: {},
        medium: {
            chargeDistanceMax: 500,
            // gravityStrength: 1,
            // nodeJitter: 0.66,
        },
        large: {
            chargeDistanceMax: 100,
            // gravityStrength: 1,
            // nodeJitter: 0.42,
        },
    },
};

const SIMULATION_WORLD = {
    velocityDecay: 0.05,
    alphaDecay: 0.001,
    useBouncyBounds: true,
    boundaryBounce: 5,
    minWorldSize: 1000,
    maxWorldSize: 1200000,
    worldScale: 400,
    boundaryPadding: 0,
    renderIntervalMs: 16,
    width: 800,
    height: 600,
    scaling: {
        small: {},
        medium: {
            // velocityDecay: 2,
            // alphaDecay: 2,
            // worldScale: 0.5,
            // boundaryPadding: 0.66,
        },
        large: {
            // velocityDecay: 10,
            // alphaDecay: 10,
            // worldScale: 0.25,
            // boundaryPadding: 0.1,
        },
    },
};

const SIMULATION_PROFILE = {
    mediumGraphThreshold: 9000,
    largeGraphThreshold: 20000,
    collisionMaxNodes: 1800,
    labelsMaxNodes: 900,
    tooltipsMaxNodes: 12000,
    pointerHitsMaxNodes: 50000,
    drawLinksMaxLinks: 1800000,
    progressiveRevealMinNodes: 500,
    perLinkOpacityMaxLinks: 12000,
    linkGlobalOpacityLarge: 0.1,
    linkGlobalOpacityMedium: 0.2,
    linkVisibilityZoomThresholdLarge: 0.001,
    linkVisibilityZoomThresholdMedium: 0.001,
    linkVisibilityZoomThresholdSmall: 0,
    renderIntervalMsLarge: 48,
    renderIntervalMsMedium: 32,
    simulationLinkStrideDivisorLarge: 60000,
    simulationLinkStrideDivisorMedium: 45000,
    constraintPhasesDivisorLarge: 5000,
    constraintPhasesDivisorMedium: 9000,
    startPausedThreshold: 11000,
    linkWorkWeight: 1.5,
};

const MAP_ZOOM = {
    labelThreshold: 0.25,
    arrowThreshold: 0.15,
    borderThreshold: 0.2,
    linkThreshold: 0.05,
    labelCap: 500,
    arrowCap: 2000,
    focusScale: 1.5,
    focusDuration: 750,
    fitPadding: 0.8,
    maxFitScale: 4,
    fallbackScale: 0.05,
    minScale: 0.001,
};

const MAP_NODES = {
    baseRadius: 3,
    radiusMultiplier: 2,
    minScreenRadius: 0.8,
    labelTruncateLength: 8,
    colorDomain: [0, 0.45, 0.5, 0.54, 1],
    colorRange: ["#ef4444", "#facc15", "#facc15", "#facc15", "#22c55e"],
    defaultOpacity: 0.3,
    borderStroke: "rgba(255, 255, 255, 0.8)",
    borderLineWidth: 0.9,
    selectedStroke: "#f472b6",
    selectedLineWidth: 2,
};

const MAP_WORLD_BOUNDS = {
    strokeColor: "rgba(139, 92, 246, 0.8)",
    lineWidth: 2,
    dashArray: [12, 6],
    padding: 150,
};

const MAP_DETAIL = {
    viewportPadding: 24,
    labelFontSizeMin: 6,
    labelFontSizeMax: 16,
    labelFontSizeMultiplier: 1.6,
    labelOffset: 3,
    labelViewportPadding: 20,
    nodeBorderViewportPadding: 1,
    hitTestPadding: 10,
    fallbackRadius: 5,
};
