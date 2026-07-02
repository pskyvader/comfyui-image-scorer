globalThis.ChainMapUI.prototype._listen = function () {
    if (this.refreshBtn) {
        this.refreshBtn.onclick = () => this.loadData();
    }
    if (this.resetViewBtn) {
        this.resetViewBtn.onclick = () => this.resetView();
    }

    const hFI = (mn, mx, mnV, mxV, cfg, isM) => {
        let mp = parseInt(mn.value);
        let xp = parseInt(mx.value);
        if (mp > xp) {
            if (isM) {
                mx.value = mp;
            } else {
                mn.value = xp;
            }
            mp = parseInt(mn.value);
            xp = parseInt(mx.value);
        }
        if (mnV) {
            mnV.textContent = globalThis.sliderToValue(mp, cfg);
        }
        if (mxV) {
            mxV.textContent = xp >= cfg.steps ? "Max" : globalThis.sliderToValue(xp, cfg);
        }
        this.saveFilters();
    };
    const bF = (mn, mx, mnV, mxV, cfg) => {
        if (mn) {
            mn.oninput = () => hFI(mn, mx, mnV, mxV, cfg, 1);
            mn.onchange = () => {
                this.saveFilters();
                this._restartSim();
            };
        }
        if (mx) {
            mx.oninput = () => hFI(mn, mx, mnV, mxV, cfg, 0);
            mx.onchange = () => {
                this.saveFilters();
                this._restartSim();
            };
        }
    };
    bF(this.minCompFilter, this.maxCompFilter, this.minCompVal, this.maxCompVal, globalThis.SLIDER.comp);
    bF(this.minChainFilter, this.maxChainFilter, this.minChainVal, this.maxChainVal, globalThis.SLIDER.chain);
    bF(this.minCompCountFilter, this.maxCompCountFilter, this.minCompCountVal, this.maxCompCountVal, globalThis.SLIDER.compCount);
    if (this.nodeTypeFilter) {
        this.nodeTypeFilter.onchange = () => {
            this.saveFilters();
            this._restartSim();
        };
    }

    if (this.nodePowerEl) {
        this.nodePowerEl.oninput = () => {
            const v = globalThis.sliderToValue(parseInt(this.nodePowerEl.value), globalThis.SLIDER.nodePower);
            if (this.nodePowerVal) {
                this.nodePowerVal.textContent = v.toFixed(2);
            }
            this._nodePower = v;
            this.saveFilters();
        };
        this.nodePowerEl.onchange = () => {
            if (this._simNodes.length) {
                this._restartSim();
            }
        };
    }

    if (this.maxNodesFilter) {
        this.maxNodesFilter.oninput = () => {
            const v = globalThis.sliderToValue(parseInt(this.maxNodesFilter.value), globalThis.SLIDER.maxNodes);
            if (this.maxNodesVal) {
                this.maxNodesVal.textContent = v >= globalThis.SLIDER.maxNodes.max ? "All" : v;
            }
            this._maxNodes = v;
            this.saveFilters();
        };
        this.maxNodesFilter.onchange = () => {
            this.saveFilters();
            this._restartSim();
        };
    }

    if (this.compareSelectedBtn) {
        this.compareSelectedBtn.onclick = () => this.compareSelected();
    }

    const tgl = (id, cb) => {
        const el = document.getElementById(id);
        if (el) {
            el.onclick = cb;
        }
    };
    tgl("toggle-labels-btn", () => {
        this._showLabels = !this._showLabels;
        this._sync();
    });
    tgl("toggle-main-chains-btn", () => {
        this._showMain = !this._showMain;
        if (this.renderer) {
            this.renderer.setLinkVisibility(this._showMain, this._showSecondary, this._showRegular);
        }
        this._sync();
    });
    tgl("toggle-secondary-chains-btn", () => {
        this._showSecondary = !this._showSecondary;
        if (this.renderer) {
            this.renderer.setLinkVisibility(this._showMain, this._showSecondary, this._showRegular);
        }
        this._sync();
    });
    tgl("toggle-regular-links-btn", () => {
        this._showRegular = !this._showRegular;
        if (this.renderer) {
            this.renderer.setLinkVisibility(this._showMain, this._showSecondary, this._showRegular);
        }
        this._sync();
    });

    const accTgl = (id, flag, onChange) => {
        const el = document.getElementById(id);
        if (!el) {
            return;
        }
        const update = () => {
            const on = this[flag];
            el.textContent = on ? "Enabled" : "Disabled";
            el.classList.toggle("text-cyan-400", on);
            el.classList.toggle("text-gray-500", !on);
            el.classList.toggle("border-cyan-500/30", on);
            el.classList.toggle("border-white/10", !on);
        };
        el.onclick = () => {
            this[flag] = !this[flag];
            update();
            this.saveFilters();
            if (onChange) {
                onChange(this[flag]);
            }
            if (this._simNodes.length && this._paused && this.renderer) {
                this._tick();
                this.renderer.renderOne();
            }
        };
        update();
    };
    accTgl("toggle-main-links-physics", "_mainLinkPhysics", () => {
        this._initActiveForces();
    });
    accTgl("toggle-secondary-physics-btn", "_secondaryChainPhysics", () => {
        this._initActiveForces();
    });
    accTgl("toggle-buoyancy", "_buoyancyEnabled", () => {
        this._initActiveForces();
    });
    accTgl("toggle-repulsion", "_repulsionEnabled", () => {
        this._initActiveForces();
    });
    accTgl("toggle-damping", "_dampingEnabled");
    accTgl("toggle-collisions", "_collisionsEnabled", () => {
        this._initActiveForces();
    });

    if (this.toggleWebglBtn) {
        const updateWebgl = (on) => {
            this._useWebGLPhysics = on;
            this.toggleWebglBtn.textContent = on ? "WebGL" : "CPU";
            this.toggleWebglBtn.classList.toggle("text-cyan-400", on);
            this.toggleWebglBtn.classList.toggle("text-gray-500", !on);
            this.toggleWebglBtn.classList.toggle("border-cyan-500/30", on);
            this.toggleWebglBtn.classList.toggle("border-white/10", !on);
            if (!on) {
                this._webglToastShown = false;
            }
            this.saveFilters();
        };
        this._updateWebglBtn = updateWebgl;
        this.toggleWebglBtn.onclick = () => {
            if (this._useWebGLPhysics) {
                updateWebgl(false);
                if (this._webglPhysics) {
                    this._webglPhysics.destroy();
                    this._webglPhysics = null;
                }
                this._simAlpha = 1;
                return;
            }
            // Test basic WebGL 2.0 + EXT_color_buffer_float availability
            const canvas = document.createElement("canvas");
            canvas.width = 1;
            canvas.height = 1;
            const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, premultipliedAlpha: false });
            if (!gl) {
                globalThis.showError("WebGL 2.0 not available in this browser.");
                return;
            }
            if (!gl.getExtension("EXT_color_buffer_float")) {
                globalThis.showError("WebGL 2.0 EXT_color_buffer_float not supported on this device.");
                gl.getExtension("WEBGL_lose_context")?.loseContext();
                return;
            }
            gl.getExtension("WEBGL_lose_context")?.loseContext();
            if (this._webglPhysics) {
                this._webglPhysics.destroy();
                this._webglPhysics = null;
            }
            this._webglNeedsReinit = true;
            this._simAlpha = 1;
            updateWebgl(true);
        };
        updateWebgl(this._useWebGLPhysics);
    }

    tgl("zoom-in-btn", () => {
        const s = this.renderer;
        if (!s || !s.camera) {
            return;
        }
        s.camera.zoom = Math.min(s.camera.zoom * 1.5, 50);
        s.camera.updateProjectionMatrix();
        s._updateNodeSizes();
        s._updateGrids();
        s.renderer.render(s.scene, s.camera);
    });
    tgl("zoom-out-btn", () => {
        const s = this.renderer;
        if (!s || !s.camera) {
            return;
        }
        s.camera.zoom = Math.max(s.camera.zoom / 1.5, 0.01);
        s.camera.updateProjectionMatrix();
        s._updateNodeSizes();
        s._updateGrids();
        s.renderer.render(s.scene, s.camera);
    });
    tgl("rotate-btn", () => {
        const s = this.renderer;
        if (!s) {
            return;
        }
        s._rotationEnabled = !s._rotationEnabled;
        s._updateRotationMode(s._rotationEnabled);
        const rotateIconOff = document.getElementById("rotate-icon-off");
        const rotateIconOn = document.getElementById("rotate-icon-on");
        if (rotateIconOff) {
            rotateIconOff.classList.toggle("hidden", s._rotationEnabled);
        }
        if (rotateIconOn) {
            rotateIconOn.classList.toggle("hidden", !s._rotationEnabled);
        }
        if (!s._rotationEnabled) {
            const controlsTarget = s.controls.target;
            s.camera.position.set(controlsTarget.x, controlsTarget.y, 1000);
            s.controls.update();
            s._updateNodeSizes();
            s._updateGrids();
            s.renderer.render(s.scene, s.camera);
        }
    });

    const pBtn = document.getElementById("play-pause-btn");
    if (pBtn) {
        pBtn.onclick = () => {
            this._paused = !this._paused;
            const pi = document.getElementById("pause-icon");
            const pl = document.getElementById("play-icon");
            if (pi) {
                pi.classList.toggle("hidden", this._paused);
            }
            if (pl) {
                pl.classList.toggle("hidden", !this._paused);
            }
            if (!this._paused && this._simNodes.length) {
                this._startSim();
            } else if (this._paused) {
                this._stopSim();
            }
        };
    }

    const physApply = (el, valEl, key, cfg, suffix) => {
        if (!el) {
            return;
        }
        el.oninput = () => {
            const v = globalThis.physicsSliderToValue(parseInt(el.value), cfg);
            if (valEl) {
                const p = cfg.precision > 0 ? cfg.precision : 0;
                valEl.textContent = (p > 0 ? v.toFixed(p) : v) + (suffix || "");
            }
            this._physicsCfg[key] = v;
            this.saveFilters();
        };
    };
    physApply(this.physBaseLinkLength, this.physBaseLinkLengthVal, "baseLinkLength", globalThis.PHYSICS_SLIDER.baseLinkLength, "px");
    physApply(this.physLinkScoreMult, this.physLinkScoreMultVal, "linkScoreMultiplier", globalThis.PHYSICS_SLIDER.linkScoreMultiplier, "x");
    physApply(this.physLinkStrength, this.physLinkStrengthVal, "linkStrength", globalThis.PHYSICS_SLIDER.linkStrength);
    physApply(this.physSecondaryLinkStrength, this.physSecondaryLinkStrengthVal, "secondaryLinkStrength", globalThis.PHYSICS_SLIDER.linkStrength);
    physApply(this.physBuoyancy, this.physBuoyancyVal, "buoyancyStrength", globalThis.PHYSICS_SLIDER.buoyancyStrength);
    physApply(this.physRepStrength, this.physRepStrengthVal, "repulsionStrength", globalThis.PHYSICS_SLIDER.repulsionStrength);
    physApply(this.physRepRange, this.physRepRangeVal, "repulsionRange", globalThis.PHYSICS_SLIDER.repulsionRange, "px");
    physApply(this.physVelocityDecay, this.physVelocityDecayVal, "velocityDecay", globalThis.PHYSICS_SLIDER.velocityDecay);
    physApply(this.physNodeSize, this.physNodeSizeVal, "nodeBaseSize", globalThis.PHYSICS_SLIDER.nodeBaseSize, "px");
    physApply(this.physAlphaDecay, this.physAlphaDecayVal, "alphaDecay", globalThis.PHYSICS_SLIDER.alphaDecay);
    physApply(this.physAlphaMin, this.physAlphaMinVal, "alphaMin", globalThis.PHYSICS_SLIDER.alphaMin);
    physApply(this.physAreaPerNode, this.physAreaPerNodeVal, "minAreaPerNode", globalThis.PHYSICS_SLIDER.minAreaPerNode, "px²");
    physApply(this.physMaxVelocity, this.physMaxVelocityVal, "maxVelocity", globalThis.PHYSICS_SLIDER.maxVelocity, "px/tick");
    const linearSlider = (el, valEl, key, steps, min, max, suffix) => {
        if (!el) {
            return;
        }
        el.oninput = () => {
            const pos = parseInt(el.value);
            const v = pos === 0 ? min : pos >= steps ? max : Math.round(min + (pos / steps) * (max - min));
            this[key] = v;
            if (valEl) {
                valEl.textContent = v + (suffix || "");
            }
            this.saveFilters();
        };
        el.onchange = () => {
            this._tickSkipAccum = 0;
            this.saveFilters();
            this._restartSim();
        };
    };
    linearSlider(this.physForcesPerTick, this.physForcesPerTickVal, "_forcesPerTick", 4, 1, 5, "");
    if (this.physTickFreq) {
        this.physTickFreq.oninput = () => {
            const pos = parseInt(this.physTickFreq.value);
            const v = pos === 0 ? 0.1 : pos >= 9 ? 1 : Math.round((0.1 + (pos / 9) * 0.9) * 10) / 10;
            this._tickFrequency = v;
            if (this.physTickFreqVal) {
                this.physTickFreqVal.textContent = Math.round(v * 100) + "%";
            }
            this.saveFilters();
        };
        this.physTickFreq.onchange = () => {
            this._tickSkipAccum = 0;
            this.saveFilters();
            this._restartSim();
        };
    }
    if (this.physAreaPerNodeVal) {
        this.physAreaPerNode.addEventListener("input", () => {
            const p = globalThis.physicsSliderToValue(parseInt(this.physAreaPerNode.value), globalThis.PHYSICS_SLIDER.minAreaPerNode);
            this.physAreaPerNodeVal.textContent = p.toLocaleString() + " px²";
        });
    }
    if (this.physAreaPerNode) {
        this.physAreaPerNode.addEventListener("change", () => {
            this._restartSim();
        });
    }
};
