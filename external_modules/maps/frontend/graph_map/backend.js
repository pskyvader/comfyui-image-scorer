/**
 * Backend Data Fetching for ChainMapUI
 */

ChainMapUI.prototype.loadData = async function() {
    if (this.loader) this.loader.classList.remove("hidden");
    try {
        const data = await api._get("/maps/graph-data");
        console.log("Backend returned " + data.nodes.length + " nodes, " + data.edges.length + " edges");
        this.rawData = data;
        this.applyFilters();
    } catch (e) {
        console.error("Failed to load graph data:", e);
        throw e;
    } finally {
        if (this.loader) this.loader.classList.add("hidden");
    }
};
