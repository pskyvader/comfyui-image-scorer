globalThis.RENDER = {
    node: {
        baseRadius: 5,
        colorDomain: [0, 0.45, 0.5, 0.54, 1],
        colorRange: ["#ef4444", "#facc15", "#facc15", "#facc15", "#22c55e"],
        highlightColor: "#f59e0b",
        selectedColor: "#f472b6",
    },
    link: {
        regularColor: "#6b7280",
        regularOpacity: 0.15,
        regularLineWidth: 0.5,
        mainChainColor: "#ffffff",
        mainChainOpacity: 0.55,
        mainChainLineWidth: 2.0,
    },
    grid: {
        levels: [10, 100, 1000, 10000, 100000],
        opacityRange: [0.015, 0.08],
    },
    border: {
        strokeColor: "rgba(139, 92, 246, 0.6)",
        lineWidth: 1.5,
        padding: 150,
    },
    label: {
        color: "rgba(226, 232, 240, 0.92)",
        cap: 500,
    },
    physics: {
        defaultBaseLinkLength: 200,
        defaultLinkScoreMultiplier: 500,
        defaultLinkStrength: 0.2,
        defaultBuoyancyStrength: 1.5,
        defaultRepulsionStrength: 0.5,
        defaultRepulsionRange: 800,
        defaultVelocityDecay: 0.08,
        defaultNodeBaseSize: 0.3,
        defaultAlphaDecay: 0.003,
        defaultAlphaMin: 0.002,
        defaultMinAreaPerNode: 40000,
    },
};

globalThis.PHYSICS_SLIDER = {
    baseLinkLength: { steps: 50, min: 10, max: 1000, precision: 0 },
    linkScoreMultiplier: { steps: 50, min: 0, max: 20, precision: 2 },
    linkStrength: { steps: 50, min: 0, max: 1, precision: 3 },
    buoyancyStrength: { steps: 50, min: 1, max: 5, precision: 2 },
    repulsionStrength: { steps: 50, min: 0, max: 20, precision: 2 },
    repulsionRange: { steps: 50, min: 1, max: 3000, precision: 0 },
    velocityDecay: { steps: 50, min: 0, max: 0.5, precision: 3 },
    nodeBaseSize: { steps: 50, min: 0.1, max: 50, precision: 1 },
    alphaDecay: { steps: 50, min: 0, max: 0.005, precision: 4 },
    alphaMin: { steps: 50, min: 0, max: 1, precision: 3 },
    minAreaPerNode: { steps: 50, min: 1000, max: 1000000, precision: 0 },
    maxVelocity: { steps: 50, min: 1, max: 500, precision: 0 },
    forcesPerTick: { steps: 4, min: 1, max: 5, precision: 0 },
    tickFrequency: { steps: 9, min: 0.1, max: 1, precision: 1 },
};

globalThis.SLIDER = {
    comp: { steps: 25, min: 1, max: 30000 },
    chain: { steps: 25, min: 1, max: 5000 },
    compCount: { steps: 25, min: 0, max: 5000 },
    nodePower: { steps: 25, min: 0, max: 2, precision: 2 },
    maxNodes: { steps: 50, min: 0, max: 50000 },
};
