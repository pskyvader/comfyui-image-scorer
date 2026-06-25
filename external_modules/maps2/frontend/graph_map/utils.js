globalThis.showError = function (message) {
    let container = document.getElementById("error-container");
    if (!container) {
        container = document.createElement("div");
        container.id = "error-container";
        container.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:99999;display:flex;flex-direction:column;gap:4px;padding:8px;pointer-events:none";
        document.body.appendChild(container);
    }
    const el = document.createElement("div");
    el.style.cssText = "background:#b91c1c;color:white;padding:8px 16px;border-radius:6px;font-size:13px;font-family:sans-serif;box-shadow:0 4px 12px rgba(0,0,0,0.3);pointer-events:auto;cursor:pointer;max-width:600px;margin:0 auto";
    el.textContent = message;
    el.onclick = function () {
        el.remove();
    };
    container.appendChild(el);
    setTimeout(function () {
        if (el.parentNode) {
            el.remove();
        }
    }, 8000);
};

globalThis.sliderToValue = function (pos, cfg) {
    if (pos <= 0) {
        return cfg.min;
    }
    if (pos >= cfg.steps) {
        return cfg.max;
    }
    const t = pos / cfg.steps;
    const range = cfg.max - cfg.min;
    if (cfg.precision !== undefined) {
        const v = cfg.min + range * Math.pow(t, 3);
        return cfg.precision > 0 ? parseFloat(v.toFixed(cfg.precision)) : Math.round(v);
    }
    return Math.round(cfg.min + range * Math.pow(t, 3));
};

globalThis.valueToSlider = function (value, cfg) {
    if (value <= cfg.min) {
        return 0;
    }
    if (value >= cfg.max) {
        return cfg.steps;
    }
    const t = Math.pow((value - cfg.min) / (cfg.max - cfg.min), 1 / 3);
    return Math.round(t * cfg.steps);
};

globalThis.physicsSliderToValue = function (pos, cfg) {
    if (pos <= 0) {
        return cfg.min;
    }
    if (pos >= cfg.steps) {
        return cfg.max;
    }
    const t = pos / cfg.steps;
    const r = cfg.max - cfg.min;
    const v = cfg.min + r * Math.pow(t, 3);
    return cfg.precision > 0 ? parseFloat(v.toFixed(cfg.precision)) : Math.round(v);
};

globalThis.valueToPhysicsSlider = function (value, cfg) {
    if (value <= cfg.min) {
        return 0;
    }
    if (value >= cfg.max) {
        return cfg.steps;
    }
    const t = Math.pow((value - cfg.min) / (cfg.max - cfg.min), 1 / 3);
    return Math.round(t * cfg.steps);
};
