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
        linkColor: "#ffffff",
        mainChainColor: "#ffffff",
        regularChainColor: "#888888",
        linkLineWidth: 1,
        linkViewportPadding: 10,
        linkOpacityMinDist: 10,
        linkOpacityMaxDist: 100000,
        linkOpacityMin: 0.1,
        linkOpacityMax: 1.0,
        linkVisibilityZoomThreshold: 0.01,
        smallLinkVisibilityZoomThresholdMultiplier: 0,
        largeLinkVisibilityZoomThresholdMultiplier: 0.01,
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
        hitTestPadding: 8, // makes clicking small nodes easier (canvas_renderer.js:130)
        mobileHitTestPadding: 4, // smaller hit area on mobile
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
// CONTROLS — user interaction settings
// Used in: simulation.js, canvas_renderer.js, main.js, controls.js, interactions.js
// =============================================
const CONTROLS = {
    pausedLinkedDragInfluence: 10,
    transitionDuration: 750,
    enablePointerHits: true,
};
