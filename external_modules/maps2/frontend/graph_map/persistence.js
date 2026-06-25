globalThis.ChainMapUI.prototype.loadFiltersFromStorage = function () {
    const ld = (el, vl, key, cfg, isMin) => {
        if (!el) {
            return;
        }
        const sv = localStorage.getItem(key);
        const v = sv !== null ? parseFloat(sv) : (isMin ? cfg.min : cfg.max);
        el.value = globalThis.valueToSlider(v, cfg);
        if (vl) {
            vl.textContent = v >= cfg.max && !isMin ? "Max" : v;
        }
    };
    ld(this.minCompFilter, this.minCompVal, "cm2_minComp", globalThis.SLIDER.comp, true);
    ld(this.maxCompFilter, this.maxCompVal, "cm2_maxComp", globalThis.SLIDER.comp, false);
    ld(this.minChainFilter, this.minChainVal, "cm2_minChain", globalThis.SLIDER.chain, true);
    ld(this.maxChainFilter, this.maxChainVal, "cm2_maxChain", globalThis.SLIDER.chain, false);
    ld(this.minCompCountFilter, this.minCompCountVal, "cm2_minCompCount", globalThis.SLIDER.compCount, true);
    ld(this.maxCompCountFilter, this.maxCompCountVal, "cm2_maxCompCount", globalThis.SLIDER.compCount, false);
    if (this.nodeTypeFilter) {
        const nodeTypeValue = localStorage.getItem("cm2_nodeType");
        if (nodeTypeValue) {
            this.nodeTypeFilter.value = nodeTypeValue;
        }
    }
    if (this.nodePowerEl) {
        const storedPower = localStorage.getItem("cm2_nodePower");
        if (storedPower !== null) {
            this.nodePowerEl.value = globalThis.valueToSlider(parseFloat(storedPower), globalThis.SLIDER.nodePower);
        }
        const currentPower = globalThis.sliderToValue(parseInt(this.nodePowerEl.value), globalThis.SLIDER.nodePower);
        if (this.nodePowerVal) {
            this.nodePowerVal.textContent = currentPower.toFixed(2);
        }
        this._nodePower = currentPower;
    }
    if (this.maxNodesFilter) {
        const storedMax = localStorage.getItem("cm2_maxNodes");
        const maxVal = storedMax !== null ? parseFloat(storedMax) : globalThis.SLIDER.maxNodes.max;
        this.maxNodesFilter.value = globalThis.valueToSlider(maxVal, globalThis.SLIDER.maxNodes);
        const resolved = globalThis.sliderToValue(parseInt(this.maxNodesFilter.value), globalThis.SLIDER.maxNodes);
        if (this.maxNodesVal) {
            this.maxNodesVal.textContent = resolved >= globalThis.SLIDER.maxNodes.max ? "All" : resolved;
        }
        this._maxNodes = resolved;
    }
    const ldPhys = (el, vl, storeKey, cfgKey, cfg, suffix) => {
        if (!el) {
            return;
        }
        const stored = localStorage.getItem(storeKey);
        const defaultVal = this._physicsCfg[cfgKey];
        const loadedVal = stored !== null ? parseFloat(stored) : (defaultVal !== undefined ? defaultVal : cfg.min);
        el.value = globalThis.valueToPhysicsSlider(loadedVal, cfg);
        const resolvedValue = globalThis.physicsSliderToValue(parseInt(el.value), cfg);
        if (vl) {
            const precision = cfg.precision > 0 ? cfg.precision : 0;
            vl.textContent = (precision > 0 ? resolvedValue.toFixed(precision) : resolvedValue) + (suffix || "");
        }
        this._physicsCfg[cfgKey] = resolvedValue;
    };
    ldPhys(this.physBaseLinkLength, this.physBaseLinkLengthVal, "cm2_physBaseLL", "baseLinkLength", globalThis.PHYSICS_SLIDER.baseLinkLength, "px");
    ldPhys(this.physLinkScoreMult, this.physLinkScoreMultVal, "cm2_physLinkScoreMult", "linkScoreMultiplier", globalThis.PHYSICS_SLIDER.linkScoreMultiplier, "x");
    ldPhys(this.physLinkStrength, this.physLinkStrengthVal, "cm2_physLinkStr", "linkStrength", globalThis.PHYSICS_SLIDER.linkStrength);
    ldPhys(this.physSecondaryLinkStrength, this.physSecondaryLinkStrengthVal, "cm2_physSecondaryLinkStr", "secondaryLinkStrength", globalThis.PHYSICS_SLIDER.linkStrength);
    ldPhys(this.physBuoyancy, this.physBuoyancyVal, "cm2_physBuoy", "buoyancyStrength", globalThis.PHYSICS_SLIDER.buoyancyStrength);
    ldPhys(this.physRepStrength, this.physRepStrengthVal, "cm2_physRepStr", "repulsionStrength", globalThis.PHYSICS_SLIDER.repulsionStrength);
    ldPhys(this.physRepRange, this.physRepRangeVal, "cm2_physRepRng", "repulsionRange", globalThis.PHYSICS_SLIDER.repulsionRange, "px");
    ldPhys(this.physVelocityDecay, this.physVelocityDecayVal, "cm2_physVelDecay", "velocityDecay", globalThis.PHYSICS_SLIDER.velocityDecay);
    ldPhys(this.physNodeSize, this.physNodeSizeVal, "cm2_physNodeSize", "nodeBaseSize", globalThis.PHYSICS_SLIDER.nodeBaseSize, "px");
    ldPhys(this.physAlphaDecay, this.physAlphaDecayVal, "cm2_physAlphaDecay", "alphaDecay", globalThis.PHYSICS_SLIDER.alphaDecay);
    ldPhys(this.physAlphaMin, this.physAlphaMinVal, "cm2_physAlphaMin", "alphaMin", globalThis.PHYSICS_SLIDER.alphaMin);
    ldPhys(this.physAreaPerNode, this.physAreaPerNodeVal, "cm2_physAreaPerNode", "minAreaPerNode", globalThis.PHYSICS_SLIDER.minAreaPerNode, "px²");
    ldPhys(this.physMaxVelocity, this.physMaxVelocityVal, "cm2_physMaxVelocity", "maxVelocity", globalThis.PHYSICS_SLIDER.maxVelocity, "px/tick");
    if (this.physAreaPerNodeVal) {
        const areaValue = globalThis.physicsSliderToValue(parseInt(this.physAreaPerNode.value), globalThis.PHYSICS_SLIDER.minAreaPerNode);
        this.physAreaPerNodeVal.textContent = areaValue.toLocaleString() + " px²";
    }
    const loadBool = (key, flag) => {
        const v = localStorage.getItem(key);
        if (v !== null) {
            this[flag] = v === "true";
        }
    };
    loadBool("cm2_secondaryChainPhysics", "_secondaryChainPhysics");
    loadBool("cm2_mainLinkPhysics", "_mainLinkPhysics");
    loadBool("cm2_buoyancyEnabled", "_buoyancyEnabled");
    loadBool("cm2_repulsionEnabled", "_repulsionEnabled");
    loadBool("cm2_dampingEnabled", "_dampingEnabled");
    loadBool("cm2_collisionsEnabled", "_collisionsEnabled");
};

globalThis.ChainMapUI.prototype.saveFilters = function () {
    const sv = (key, value) => localStorage.setItem(key, String(value));
    sv("cm2_minComp", globalThis.sliderToValue(parseInt(this.minCompFilter.value), globalThis.SLIDER.comp));
    sv("cm2_maxComp", globalThis.sliderToValue(parseInt(this.maxCompFilter.value), globalThis.SLIDER.comp));
    sv("cm2_minChain", globalThis.sliderToValue(parseInt(this.minChainFilter.value), globalThis.SLIDER.chain));
    sv("cm2_maxChain", globalThis.sliderToValue(parseInt(this.maxChainFilter.value), globalThis.SLIDER.chain));
    sv("cm2_minCompCount", globalThis.sliderToValue(parseInt(this.minCompCountFilter.value), globalThis.SLIDER.compCount));
    sv("cm2_maxCompCount", globalThis.sliderToValue(parseInt(this.maxCompCountFilter.value), globalThis.SLIDER.compCount));
    if (this.nodeTypeFilter) {
        sv("cm2_nodeType", this.nodeTypeFilter.value);
    }
    if (this.nodePowerEl) {
        sv("cm2_nodePower", globalThis.sliderToValue(parseInt(this.nodePowerEl.value), globalThis.SLIDER.nodePower));
    }
    sv("cm2_maxNodes", this._maxNodes);
    sv("cm2_physBaseLL", this._physicsCfg.baseLinkLength);
    sv("cm2_physLinkScoreMult", this._physicsCfg.linkScoreMultiplier);
    sv("cm2_physLinkStr", this._physicsCfg.linkStrength);
    sv("cm2_physSecondaryLinkStr", this._physicsCfg.secondaryLinkStrength);
    sv("cm2_physBuoy", this._physicsCfg.buoyancyStrength);
    sv("cm2_physRepStr", this._physicsCfg.repulsionStrength);
    sv("cm2_physRepRng", this._physicsCfg.repulsionRange);
    sv("cm2_physVelDecay", this._physicsCfg.velocityDecay);
    sv("cm2_physNodeSize", this._physicsCfg.nodeBaseSize);
    sv("cm2_physAlphaDecay", this._physicsCfg.alphaDecay);
    sv("cm2_physAlphaMin", this._physicsCfg.alphaMin);
    sv("cm2_physMaxVelocity", this._physicsCfg.maxVelocity);
    sv("cm2_physAreaPerNode", this._physicsCfg.minAreaPerNode);
    sv("cm2_secondaryChainPhysics", this._secondaryChainPhysics);
    sv("cm2_mainLinkPhysics", this._mainLinkPhysics);
    sv("cm2_buoyancyEnabled", this._buoyancyEnabled);
    sv("cm2_repulsionEnabled", this._repulsionEnabled);
    sv("cm2_dampingEnabled", this._dampingEnabled);
    sv("cm2_collisionsEnabled", this._collisionsEnabled);
};
