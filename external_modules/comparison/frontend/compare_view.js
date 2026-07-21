/**
 * Comparison Mode — Phase-Driven View
 *
 * All rendering of the per-image detail, footer, phase badge, and debug panel
 * lives here. The backend sends a flat payload:
 *   {
 *     left:  describe_image(node_a),
 *     right: describe_image(node_b),
 *     pair:  describe_pair(node_a, node_b, phase_index)
 *   }
 * The phase index is an int (0=seed, 1=anchor, 2=collapsible, 3=chain_merge,
 * 4=refine, 5=fallback); its human label is mapped below.
 */

const CompareView = (() => {
    const _baseLogger = window.FrontendLogger || { create: () => console };
    const viewLogger = _baseLogger.create("external_modules.comparison.frontend.compare_view");

    let _phases = [];

    function setPhases(phaseList) {
        _phases = phaseList;
    }

    function phaseLabel(phaseIndex) {
        const p = _phases[phaseIndex];
        if (p && p.phase_label) {
            return p.phase_label;
        }
        return `Phase ${phaseIndex}`;
    }

    function phaseCardClass(phaseIndex) {
        const p = _phases[phaseIndex];
        if (p && p.card_class) {
            return p.card_class;
        }
        return "card-fallback";
    }

    // Per-image detail is phase-driven. What to show is decided by phase + data.
    function detailLines(node, phaseIndex) {
        const lines = [];
        // Score + comparisons are always relevant.
        lines.push(`Score: ${Utils.formatScore(node.score)}`);
        lines.push(`Comparisons: ${node.comparison_count}`);
        // Later phases surface structural context.
        const phase = _phases[phaseIndex];
        if (phase && phase.show_chain_info) {
            if (node.chain_length > 0) {
                lines.push(`Chain ${node.chain_id}:${node.chain_length} (main-chain of ${node.chain_main_members})`);
            }
            if (node.is_top) {
                lines.push(`Top`);
            } else if (node.is_bottom) {
                lines.push(`Bottom`);
            }
        }
        if (phase && phase.show_mu_sigma) {
            lines.push(`μ: ${node.rating_mu.toFixed(2)}`);
            lines.push(`σ: ${node.rating_sigma.toFixed(2)}`);
        }
        return lines.join(" | ");
    }

    function renderStatsLine(el, text) {
        if (el) {
            el.textContent = text;
        }
    }

    function renderSeedBadge(filenameEl, isSeed) {
        const parent = filenameEl.parentElement;
        let badge = parent.querySelector(".seed-badge");
        if (isSeed) {
            if (!badge) {
                badge = document.createElement("span");
                badge.className = "seed-badge";
                badge.textContent = "🌱";
                parent.appendChild(badge);
            }
        } else if (badge) {
            badge.remove();
        }
    }

    function renderPhaseCards(leftCard, rightCard, phaseIndex) {
        const allCardClasses = _phases.map(p => p.card_class).concat(["collapsible-card"]);
        leftCard.classList.remove(...allCardClasses);
        rightCard.classList.remove(...allCardClasses);
        const cls = phaseCardClass(phaseIndex);
        leftCard.classList.add(cls);
        rightCard.classList.add(cls);
    }

    function renderFooter(pair, left, right) {
        const footer = document.getElementById("global-stats-footer");
        if (!footer) {
            return;
        }
        // JSON serializes dict keys as strings; normalize level counts to int keys.
        const level = {};
        for (const [k, v] of Object.entries(pair.level)) {
            level[Number(k)] = v;
        }

        // An image at or above the threshold is done (has enough comparisons).
        // So "remaining" is strictly below threshold and "ready" is at or above.
        const countRemaining = (threshold) => {
            let n = 0;
            for (const [lvl, c] of Object.entries(level)) {
                if (Number(lvl) < threshold) {
                    n += c;
                }
            }
            return n;
        };
        const countReady = (threshold) => {
            let n = 0;
            for (const [lvl, c] of Object.entries(level)) {
                if (Number(lvl) >= threshold) {
                    n += c;
                }
            }
            return n;
        };

        let line;
        const phaseName = _phases[pair.phase]?.name;
        switch (phaseName) {
            case "seed": {
                // Seed bootstrap: progress toward seed_target comparisons.
                const remaining = countRemaining(pair.seed_target_comparisons);
                const ready = countReady(pair.seed_target_comparisons);
                line = `Bootstrap seed - ${remaining.toLocaleString()} images below ${pair.seed_target_comparisons} comparisons - ${ready.toLocaleString()} at \u2265${pair.seed_target_comparisons} - ${pair.total_images.toLocaleString()} images - ${pair.total_comparisons.toLocaleString()} comparisons`;
                break;
            }
            case "anchor": {
                // Anchor insert: integrate low-count images up to insertion_target.
                const remaining = countRemaining(pair.insertion_target_comparisons);
                const ready = countReady(pair.insertion_target_comparisons);
                line = `Anchor insert - ${remaining.toLocaleString()} images below ${pair.insertion_target_comparisons} comparisons - ${ready.toLocaleString()} at \u2265${pair.insertion_target_comparisons} - ${pair.total_images.toLocaleString()} images - ${pair.total_comparisons.toLocaleString()} comparisons`;
                break;
            }
            case "collapsible": {
                // Collapsible: by definition both images share one component, so the
                // component extremes are the merge frontier. top/bottom are "in
                // progress"; everything else is a fully merged (ready) image.
                const compId = left.component_id;
                const compSize = left.component_size;
                const top = left._extremes.top;
                const bottom = left._extremes.bottom;
                const inProgress = top + bottom;
                const ready = pair.total_images - inProgress;
                line = `Collapsible pairs${pair.collapsable ? " (resolving a branch)" : ""} - component ${compId} (${compSize.toLocaleString()} images) - top ${top.toLocaleString()} / bottom ${bottom.toLocaleString()} - ${inProgress.toLocaleString()} in progress / ${ready.toLocaleString()} ready - ${pair.total_images.toLocaleString()} images - ${pair.total_comparisons.toLocaleString()} comparisons`;
                break;
            }
            case "chain_merge":
                line = `Chain merge - component ${left.component_id} (${left.component_size.toLocaleString()} images) - ${pair.total_chains.toLocaleString()} chains (target ${pair.target_chains}) - reserve ${pair.reserve_count} - ${pair.total_images.toLocaleString()} images - ${pair.total_comparisons.toLocaleString()} comparisons`;
                break;
            case "refine":
                line = `Uncertainty refine - ${pair.total_images.toLocaleString()} images - ${pair.total_comparisons.toLocaleString()} comparisons`;
                break;
            default:
                line = `Fallback - ${pair.total_images.toLocaleString()} images - ${pair.total_comparisons.toLocaleString()} comparisons`;
                break;
        }
        footer.textContent = line;
    }

    function renderDebug(pair, left, right, debugEl) {
        if (!debugEl) {
            return;
        }
        const probA = pair.probability_a_beats_b;
        const probB = (1 - probA).toFixed(4);
        const winnerText = `winner: left (${Math.round(probA * 100)}%)`;
        // JSON serializes dict keys as strings; normalize to int keys.
        const level = {};
        for (const [k, v] of Object.entries(pair.level)) {
            level[Number(k)] = v;
        }
        const levels = Object.keys(level).map(Number).sort((a, b) => a - b);

        // Show the first N levels explicitly (seed target or 10), group the rest.
        const cap = Math.max(10, pair.seed_target_comparisons || 10);
        const shown = levels.filter((lvl) => lvl <= cap);
        const restLevels = levels.filter((lvl) => lvl > cap);
        let levelRows = shown
            .map((lvl) => `<tr class="h-5"><td class="text-gray-500 pr-4">Level ${lvl}</td><td class="text-white text-right">${level[lvl]}</td></tr>`)
            .join("");
        if (restLevels.length > 0) {
            const restTotal = restLevels.reduce((sum, lvl) => sum + level[lvl], 0);
            levelRows += `<tr class="h-5"><td class="text-gray-500 pr-4">Level ${cap + 1}+</td><td class="text-white text-right">${restTotal}</td></tr>`;
        }

        const sameComponent = pair.same_component
            ? `yes (component ${left.component_id}, ${left.component_size} images)`
            : "no";

        const nodeRows = (node) => `
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Score</td><td class="text-white text-right font-mono">${Utils.formatScore(node.score)}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Skill (μ)</td><td class="text-white text-right">${node.rating_mu}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Uncertainty (σ)</td><td class="text-white text-right">${node.rating_sigma}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Comparisons</td><td class="text-white text-right">${node.comparison_count}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Chain (id:size)</td><td class="text-white text-right">${node.chain_id}:${node.chain_length}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Chain Main Members</td><td class="text-white text-right">${node.chain_main_members}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Component Size</td><td class="text-white text-right">${node.component_size}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Component ID</td><td class="text-white text-right">${node.component_id}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Is Seed</td><td class="text-white text-right">${node.is_seed ? "yes" : "no"}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Is Top</td><td class="text-white text-right">${node.is_top ? "yes" : "no"}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Is Bottom</td><td class="text-white text-right">${node.is_bottom ? "yes" : "no"}</td></tr>
            <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Extremes</td><td class="text-white text-right">top ${node._extremes.top} / bottom ${node._extremes.bottom}</td></tr>
        `;

        debugEl.innerHTML = `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8 w-full items-start">
                <table class="w-full text-left border-collapse">
                    <thead><tr><th colspan="2" class="text-purple-400 border-b border-purple-500/20 pb-2 mb-2 uppercase tracking-widest text-[9px]">Selection Logic</th></tr></thead>
                    <tbody class="text-[10px]">
                        <tr class="h-6"><td class="text-gray-500 pr-4">Phase</td><td class="text-white font-bold">${phaseLabel(pair.phase)}</td></tr>
                        <tr class="h-6"><td class="text-gray-500 pr-4">Collapsible</td><td class="text-white">${pair.collapsable ? "yes" : "no"}</td></tr>
                        <tr class="h-6"><td class="text-gray-500 pr-4">Same Component</td><td class="text-white">${sameComponent}</td></tr>
                        <tr class="h-6"><td class="text-gray-500 pr-4">${winnerText} / right (${Math.round(probB * 100)}%)</td><td class="text-white text-right">left ${probA.toFixed(4)}</td></tr>
                    </tbody>
                </table>
                <table class="w-full text-left border-collapse">
                    <thead><tr><th colspan="2" class="text-purple-400 border-b border-purple-500/20 pb-2 mb-2 uppercase tracking-widest text-[9px]">Level Distribution (first ${cap + 1})</th></tr></thead>
                    <tbody class="text-[10px]">${levelRows}</tbody>
                </table>
            </div>
            <div class="col-span-1 md:col-span-2 mt-4 grid grid-cols-1 md:grid-cols-2 gap-8">
                <table class="w-full text-left border-collapse">
                    <thead><tr><th colspan="2" class="text-purple-400 border-b border-purple-500/20 pb-2 mb-2 uppercase tracking-widest text-[9px]">Pair Config</th></tr></thead>
                    <tbody class="text-[10px]">
                        <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Total Images</td><td class="text-white text-right">${pair.total_images}</td></tr>
                        <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Total Comparisons</td><td class="text-white text-right">${pair.total_comparisons}</td></tr>
                        <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Seed Target</td><td class="text-white text-right">${pair.seed_target_comparisons}</td></tr>
                        <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Seed Percentage</td><td class="text-white text-right">${pair.seed_percentage}</td></tr>
                        <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Insertion Target</td><td class="text-white text-right">${pair.insertion_target_comparisons}</td></tr>
                        <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Min Comparisons (pair)</td><td class="text-white text-right">${pair.base_level}</td></tr>
                        <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Reserve Count</td><td class="text-white text-right">${pair.reserve_count}</td></tr>
                    </tbody>
                </table>
                <table class="w-full text-left border-collapse">
                    <thead><tr><th colspan="2" class="text-purple-400 border-b border-purple-500/20 pb-2 mb-2 uppercase tracking-widest text-[9px]">Filenames</th></tr></thead>
                    <tbody class="text-[10px]">
                        <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Left</td><td class="text-white text-right">${left.filename}</td></tr>
                        <tr class="h-6 border-b border-white/5"><td class="text-gray-500 pr-4">Right</td><td class="text-white text-right">${right.filename}</td></tr>
                    </tbody>
                </table>
            </div>
            <div class="col-span-1 md:col-span-2 mt-4 grid grid-cols-1 md:grid-cols-2 gap-8">
                <table class="w-full text-left border-collapse">
                    <thead><tr><th class="text-purple-400 border-b border-purple-500/20 pb-2 mb-2 uppercase tracking-widest text-[9px]">Node Metrics (Left)</th><th class="text-[8px] text-gray-500 text-right uppercase tracking-tighter">Value</th></tr></thead>
                    <tbody class="text-[10px]">${nodeRows(left)}</tbody>
                </table>
                <table class="w-full text-left border-collapse">
                    <thead><tr><th class="text-pink-400 border-b border-pink-500/20 pb-2 mb-2 uppercase tracking-widest text-[9px]">Node Metrics (Right)</th><th class="text-[8px] text-gray-500 text-right uppercase tracking-tighter">Value</th></tr></thead>
                    <tbody class="text-[10px]">${nodeRows(right)}</tbody>
                </table>
            </div>
        `;
    }

    function render(pair, left, right, elements) {
        viewLogger.info("render payload:", null, {
            pairKeys: pair ? Object.keys(pair) : null,
            pairPhase: pair ? pair.phase : undefined,
            leftKeys: left ? Object.keys(left) : null,
            rightKeys: right ? Object.keys(right) : null,
        });
        if (!pair) {
            viewLogger.error("render called with undefined `pair` — backend payload missing `pair` envelope");
        }
        renderSeedBadge(elements.leftFilename, left.is_seed);
        renderSeedBadge(elements.rightFilename, right.is_seed);

        renderStatsLine(
            document.getElementById("left-stats-line"),
            detailLines(left, pair.phase),
        );
        renderStatsLine(
            document.getElementById("right-stats-line"),
            detailLines(right, pair.phase),
        );

        renderPhaseCards(elements.leftCard, elements.rightCard, pair.phase);
        renderFooter(pair, left, right);
        renderDebug(pair, left, right, elements.debugContent);
    }

    function renderDescriptions(labels) {
        const container = document.getElementById("phase-descriptions");
        if (!container || _phases.length === 0) {
            return;
        }
        const fmt = (v) => (v != null ? Number(v).toLocaleString() : "\u2014");
        const seedSize = fmt(labels?.seed_size);
        const seedTarget = fmt(labels?.seed_target);
        const insertionTarget = fmt(labels?.insertion_target);
        const header = '<div class="text-[9px] uppercase tracking-widest font-bold mb-2 opacity-40">Pair Selection Strategy</div>';
        const items = _phases.map((p) => {
            const cls = p.description_class || "text-gray-400";
            const label = p.phase_label || "";
            const desc = (p.description || "")
                .replace(/\{seed_size\}/g, seedSize)
                .replace(/\{seed_target\}/g, seedTarget)
                .replace(/\{insertion_target\}/g, insertionTarget);
            return `<div><span class="${cls} font-bold">${label}</span> — ${desc}</div>`;
        }).join("\n");
        container.innerHTML = header + items;
    }

    return { render, phaseLabel, phaseCardClass, setPhases, renderDescriptions };
})();

window.CompareView = CompareView;
