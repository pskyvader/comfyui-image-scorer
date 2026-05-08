/**
 * Backend Data Fetching for ChainMapUI
 */

ChainMapUI.prototype.loadData = async function() {
    if (this.loader) this.loader.classList.remove("hidden");
    try {
        // api is a global object defined in api.js
        const data = await api.getGraphData();
        this.rawData = data;
        this.applyFilters();
    } catch (e) {
        if (typeof Utils !== 'undefined') {
            Utils.showToast("Failed to load graph data", "error");
        } else {
            console.error("Failed to load graph data", e);
        }
    } finally {
        if (this.loader) this.loader.classList.add("hidden");
    }
};
