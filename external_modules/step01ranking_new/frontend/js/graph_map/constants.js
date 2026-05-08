/**
 * Graph Map Configuration Constants
 * Organized by category: Visuals, Interaction, Filters, Physics, and Scaling.
 */

const MAP_VISUALS = {
    nodeMinRadius: 3,
    highlightColor: "#f472b6",
    linkVisibilityZoomThreshold: 0.1,
    linkGlobalOpacity: 0.25,
    drawLinks: true,
    progressiveReveal: true,
    enablePointerHits: true,
};

const MAP_INTERACTION = {
    zoomExtent: [0.005, 5],
    interactThreshold: 0.01,
    transitionDuration: 750,
};

const MAP_FILTERS = {
    sliderMax: 20,
};

const SIMULATION_LINKS = {
    linkDistance: 6,
    linkStrength: 1,
    linkStrengthInitialFactor: 0.001,
    linkStrengthFinalFactor: 1,
    linkStrengthRampMs: 10000,
    linkStrengthRampCurve: 1.0,
    scoreDistanceMultiplier: 10,
    countDistanceMultiplier: 20,
    maxLinkDistance: 100000,
    linkConstraintSlack: 1.2,
    linkConstraintPull: 0.5,
    linkConstraintInitialSlackExtra: 0.9,
    scaling: {
        small: {
            linkDistance: 1,
            linkStrength: 1,
            scoreDistanceMultiplier: 1,
            countDistanceMultiplier: 1,
            maxLinkDistance: 1,
            constraintSlack: 1,
            constraintPull: 1,
            linkStrengthRampMs: 1,
        },
        medium: {
            linkDistance: 0.82,
            linkStrength: 0.9,
            scoreDistanceMultiplier: 0.75,
            countDistanceMultiplier: 0.45,
            maxLinkDistance: 0.5,
            constraintSlack: 0.92,
            constraintPull: 1.2,
            linkStrengthRampMs: 0.8,
        },
        large: {
            linkDistance: 0.68,
            linkStrength: 0.78,
            scoreDistanceMultiplier: 0.45,
            countDistanceMultiplier: 0.2,
            maxLinkDistance: 0.1,
            constraintSlack: 0.9,
            constraintPull: 1.5,
            linkStrengthRampMs: 0.55,
        },
    },
};

const SIMULATION_NODES = {
    chargeStrength: -30,
    chargeDistanceMax: 1000,
    gravityStrength: 1,
    scoreExponent: 2,
    neutralDeadZone: 0,
    collisionRadius: 18,
    nodeJitter: 84,
    pausedLinkedDragInfluence: 1,
    scaling: {
        small: {
            chargeDistanceMax: 1,
            gravityStrength: 1,
            nodeJitter: 1,
        },
        medium: {
            chargeDistanceMax: 0.24,
            gravityStrength: 1,
            nodeJitter: 0.66,
        },
        large: {
            chargeDistanceMax: 0.16,
            gravityStrength: 1,
            nodeJitter: 0.42,
        },
    },
};

const SIMULATION_WORLD = {
    velocityDecay: 0.008,
    alphaDecay: 0.00008,
    useBouncyBounds: true,
    boundaryBounce: 0.85,
    minWorldSize: 2000,
    maxWorldSize: 120000,
    worldScale: 200,
    boundaryPadding: 1,
    renderIntervalMs: 16,
    width: 800,
    height: 600,
    scaling: {
        small: {
            velocityDecay: 1,
            alphaDecay: 1,
            worldScale: 1,
            boundaryPadding: 1,
        },
        medium: {
            velocityDecay: 2,
            alphaDecay: 2,
            worldScale: 0.5,
            boundaryPadding: 0.66,
        },
        large: {
            velocityDecay: 10,
            alphaDecay: 10,
            worldScale: 0.25,
            boundaryPadding: 0.1,
        },
    },
};