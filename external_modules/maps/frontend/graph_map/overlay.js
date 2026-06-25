/**
 * UI Overlays (Tooltips, Details Panel, HUD) for ChainMapUI
 */

ChainMapUI.prototype.updateHUD = function({ k }) {
    if (this.zoomScaleEl) this.zoomScaleEl.textContent = `${(k || 1).toFixed(2)}x`;
    if (this.viewCoordsEl && this.renderer && this.renderer.subRenderer) {
        const sub = this.renderer.subRenderer;
        if (sub.controls) {
            const t = sub.controls.target;
            this.viewCoordsEl.textContent = `X: ${Math.round(t.x)}, Y: ${Math.round(t.y)}`;
        } else if (sub._fallback && sub.worldBounds) {
            const cx = sub.worldBounds.x + sub.worldBounds.width / 2;
            const cy = sub.worldBounds.y + sub.worldBounds.height / 2;
            const px = sub._panX || 0;
            const py = sub._panY || 0;
            const viewX = cx - px / (sub._tc ? sub._tc.scale : 1);
            const viewY = cy + py / (sub._tc ? sub._tc.scale : 1);
            this.viewCoordsEl.textContent = `X: ${Math.round(viewX)}, Y: ${Math.round(viewY)}`;
        }
    }
};

ChainMapUI.prototype.showTooltip = function(event, node) {
    this.tooltip.classList.remove("hidden");
    const rect = this.container.getBoundingClientRect();
    this.tooltip.style.left = `${event.clientX - rect.left + 10}px`;
    this.tooltip.style.top = `${event.clientY - rect.top + 10}px`;
    let chainHtml = '';
    if (node._chainPrev || node._chainNext) {
        chainHtml = `<div class="text-[8px] text-purple-400">` +
            (node._chainPrev ? `Chain ↑ ${node._chainPrev.split('/').pop()}` : `Chain ↑ (start)`) +
            ` | ` +
            (node._chainNext ? `Chain ↓ ${node._chainNext.split('/').pop()}` : `Chain ↓ (end)`) +
            `</div>`;
    } else {
        chainHtml = `<div class="text-[8px] text-gray-600 mt-0.5">No chain neighbors</div>`;
    }
    this.tooltip.innerHTML = `
        <div class="font-semibold truncate" style="cursor:pointer" onclick="window.chainMapUI.showNodeDetailsById('${node.id.replace(/'/g, "\\'")}')">${node.id.split('/').pop()}</div>
        <div class="text-[9px] text-gray-400">${node.score.toFixed(3)} · cmp:${node.comparison_count} · comp:${node._component_size}</div>
        ${chainHtml}
    `;
};

ChainMapUI.prototype.hideTooltip = function() {
    if (this._tooltipTimer) clearTimeout(this._tooltipTimer);
    this._tooltipTimer = setTimeout(() => {
        this.tooltip.classList.add("hidden");
    }, 200);
};

ChainMapUI.prototype.hideNodeDetails = function() {
    if (this.nodeDetails) {
        this.nodeDetails.classList.add("hidden");
    }
    this._activeDetailsNode = null;
    if (this.renderer) {
        this.renderer.setHighlightedChain(null);
    }
};

ChainMapUI.prototype.showNodeDetails = function(d) {
    try {
        if (!this.nodeDetails) return;
        this._activeDetailsNode = d;
        if (this.renderer) this.renderer.setHighlightedChain(null);
        this.nodeDetails.classList.remove("hidden");
        if (window.chainMapUI) window.chainMapUI.updateDetailsPosition();
        
        const filename = d.id.split('/').pop();
        this.nodeDetails.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <h3 class="text-white font-bold text-[10px] truncate max-w-[80%]">${filename}</h3>
                <button onclick="window.chainMapUI.hideNodeDetails()" class="text-gray-500 hover:text-white p-1">&times;</button>
            </div>
            <div class="flex flex-col gap-2">
                <img src="/images/${encodeURIComponent(d.id)}" class="w-full h-32 md:h-48 object-contain rounded bg-black/40 border border-white/5" onerror="this.src='/output/ranked/${encodeURIComponent(d.id)}'">
                
                <div class="flex flex-wrap gap-x-3 gap-y-1 text-[9px] text-gray-300 justify-center">
                    <span class="flex items-center gap-1">Score: <b class="text-purple-400">${d.score.toFixed(3)}</b></span>
                    <span class="flex items-center gap-1">Component: <b class="text-purple-400">${d._component_size}</b></span>
                    <span class="flex items-center gap-1">Comparisons: <b class="text-purple-400">${d.comparison_count}</b></span>
                </div>

                <div class="flex gap-2">
                    <button onclick="window.chainMapUI.toggleSelection('${d.id}')" class="flex-1 py-1.5 bg-pink-600 hover:bg-pink-500 rounded text-[9px] font-bold transition">
                        Select
                    </button>
                    <button onclick="window.chainMapUI.focusNode('${d.id}')" class="px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded text-[9px] transition">
                        Focus
                    </button>
                </div>
            </div>
        `;
    } catch (e) {
        console.error("showNodeDetails error:", e);
    }
};

ChainMapUI.prototype.showLinkDetails = function(link) {
    try {
        if (!this.nodeDetails) return;
        this._activeDetailsNode = null; // We might want to follow a node or something, but for links it can just be static or we can center it
        this.nodeDetails.classList.remove("hidden");
        
        // Link details panel doesn't have an _activeDetailsNode, so it won't follow. 
        // We'll reset it to a default position so it's not hidden offscreen if it was following something before.
        this.nodeDetails.classList.add('top-4', 'left-4');
        this.nodeDetails.style.left = '';
        this.nodeDetails.style.top = '';
        
        let chainInfoHtml = '';
        if (link.isMainChain && link.source._allChains && link.target._allChains) {
            // Find intersection of chains
            const sharedChains = link.source._allChains.filter(c => link.target._allChains.includes(c));
            if (sharedChains.length > 0) {
                const chainId = sharedChains[0];
                if (this.renderer) this.renderer.setHighlightedChain(chainId);
                chainInfoHtml = `
                    <div class="text-[9px] text-purple-400 mb-2 font-bold flex items-center justify-between">
                        <span>Main Chain Segment</span>
                        <span class="bg-purple-900/50 px-2 py-0.5 rounded">Chain ${chainId}</span>
                    </div>
                `;
            } else if (this.renderer) {
                this.renderer.setHighlightedChain(null);
            }
        } else if (this.renderer) {
            this.renderer.setHighlightedChain(null);
        }

        const srcName = link.source.id.split('/').pop();
        const tgtName = link.target.id.split('/').pop();
        this.nodeDetails.innerHTML = `
            <div class="flex justify-between items-start mb-2">
                <h3 class="text-white font-bold text-[10px]">Link Details</h3>
                <button onclick="window.chainMapUI.hideNodeDetails()" class="text-gray-500 hover:text-white p-1">&times;</button>
            </div>
            ${chainInfoHtml}
            <div class="flex flex-col gap-2 text-[9px] text-gray-300">
                <div class="flex justify-between items-center p-2 bg-white/5 rounded">
                    <div>
                        <b class="text-purple-400">${srcName}</b>
                        <span class="text-gray-500 ml-1">score ${link.source.score.toFixed(3)}</span>
                    </div>
                    <span class="text-gray-600">&harr;</span>
                    <div class="text-right">
                        <b class="text-purple-400">${tgtName}</b>
                        <span class="text-gray-500 ml-1">score ${link.target.score.toFixed(3)}</span>
                    </div>
                </div>
                <div class="flex gap-2">
                    <button onclick="window.chainMapUI.focusNode('${link.source.id}')" class="flex-1 py-1.5 bg-white/10 hover:bg-white/20 rounded font-bold transition">
                        Focus Source
                    </button>
                    <button onclick="window.chainMapUI.focusNode('${link.target.id}')" class="flex-1 py-1.5 bg-white/10 hover:bg-white/20 rounded font-bold transition">
                        Focus Target
                    </button>
                </div>
            </div>
        `;
    } catch (e) {
        console.error("showLinkDetails error:", e);
    }
};

ChainMapUI.prototype.showNodeDetailsById = function(id) {
    const node = this._simNodes.find(n => n.id === id);
    if (node) this.showNodeDetails(node);
};

ChainMapUI.prototype.updateSelectionUI = function() {
    const count = this.selectedNodes.length;
    if (this.selectedCountEl) this.selectedCountEl.textContent = count;
    if (this.compareSelectedBtn) this.compareSelectedBtn.classList.toggle("hidden", count < 2);
};
