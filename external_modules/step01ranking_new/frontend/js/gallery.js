// gallery.js

class GalleryApp {
    constructor() {
        this.currentPage = 1;
        this.perPage = 30; // Increased to make scrolling nicer
        this.currentFilters = {};
        this.allImages = [];
        this.currentImageIndex = -1;
        this.isLoading = false;
        this.hasMore = true;

        this.cacheElements();
        this.attachEventListeners();
        this.setupModal();
        this.setupInfiniteScroll();
        this.initialize();
    }

    cacheElements() {
        this.scoreMin = document.getElementById("score-min");
        this.scoreMax = document.getElementById("score-max");
        this.confidenceMin = document.getElementById("confidence-min");
        this.confidenceMax = document.getElementById("confidence-max");
        this.comparisonsMin = document.getElementById("comparisons-min");
        this.comparisonsMax = document.getElementById("comparisons-max");
        this.scoreDisplay = document.getElementById("score-display");
        this.confidenceDisplay = document.getElementById("confidence-display");
        this.comparisonsDisplay = document.getElementById("comparisons-display");
        this.sortFilter = document.getElementById("sort-filter");
        this.galleryStatus = document.getElementById("gallery-status");
        this.galleryGrid = document.getElementById("gallery-grid");
        this.statusBadge = document.getElementById("status-indicator");
        this.scrollSentinel = document.getElementById("scroll-sentinel");

        // Modal
        this.modal = document.getElementById("image-modal");
        this.modalContainer = document.getElementById("modal-container");
        this.modalImage = document.getElementById("modal-image");
        this.modalFilename = document.getElementById("modal-filename");
        this.modalMetadata = document.getElementById("modal-metadata");
        this.modalClose = document.querySelector(".modal-close");
        this.modalPrev = document.getElementById("modal-prev");
        this.modalNext = document.getElementById("modal-next");
        
        this.modalHistory = document.getElementById("modal-history");
        this.historyWins = document.getElementById("history-wins");
        this.historyLosses = document.getElementById("history-losses");
    }

    attachEventListeners() {
        const updateRanges = () => {
            // Ensure min <= max for each range
            if (parseFloat(this.scoreMin.value) > parseFloat(this.scoreMax.value)) {
                this.scoreMin.value = this.scoreMax.value;
            }
            if (parseFloat(this.confidenceMin.value) > parseFloat(this.confidenceMax.value)) {
                this.confidenceMin.value = this.confidenceMax.value;
            }
            if (parseInt(this.comparisonsMin.value) > parseInt(this.comparisonsMax.value)) {
                this.comparisonsMin.value = this.comparisonsMax.value;
            }
            
            // Update display values
            this.scoreDisplay.textContent = `${parseFloat(this.scoreMin.value).toFixed(2)} - ${parseFloat(this.scoreMax.value).toFixed(2)}`;
            this.confidenceDisplay.textContent = `${parseFloat(this.confidenceMin.value).toFixed(2)} - ${parseFloat(this.confidenceMax.value).toFixed(2)}`;
            const compMax = parseInt(this.comparisonsMax.value);
            this.comparisonsDisplay.textContent = `${this.comparisonsMin.value} - ${compMax >= 1000 ? '1000+' : compMax}`;
        };

        const debouncedSearch = Utils.debounce(() => this.search(), 300);
        
        this.scoreMin.addEventListener("input", () => { updateRanges(); debouncedSearch(); });
        this.scoreMax.addEventListener("input", () => { updateRanges(); debouncedSearch(); });
        this.confidenceMin.addEventListener("input", () => { updateRanges(); debouncedSearch(); });
        this.confidenceMax.addEventListener("input", () => { updateRanges(); debouncedSearch(); });
        this.comparisonsMin.addEventListener("input", () => { updateRanges(); debouncedSearch(); });
        this.comparisonsMax.addEventListener("input", () => { updateRanges(); debouncedSearch(); });
        this.sortFilter.addEventListener("change", () => this.search());
    }

    setupModal() {
        this.modalClose.addEventListener("click", () => this.closeModal());
        
        this.modal.addEventListener("click", (e) => {
            if (e.target === this.modal) this.closeModal();
        });

        this.modalPrev.addEventListener("click", (e) => {
            e.stopPropagation();
            this.showPrevImage();
        });
        
        this.modalNext.addEventListener("click", (e) => {
            e.stopPropagation();
            this.showNextImage();
        });

        document.addEventListener("keydown", (e) => {
            if (!this.modal.classList.contains("hidden")) {
                if (e.key === "Escape") this.closeModal();
                if (e.key === "ArrowLeft") this.showPrevImage();
                if (e.key === "ArrowRight") this.showNextImage();
            }
        });

        // Touch swipe support for mobile
        let touchStartX = 0;
        let touchStartY = 0;
        let touchStartTime = 0;

        this.modal.addEventListener("touchstart", (e) => {
            touchStartX = e.changedTouches[0].clientX;
            touchStartY = e.changedTouches[0].clientY;
            touchStartTime = Date.now();
        }, { passive: true });

        this.modal.addEventListener("touchend", (e) => {
            const dx = e.changedTouches[0].clientX - touchStartX;
            const dy = e.changedTouches[0].clientY - touchStartY;
            const dt = Date.now() - touchStartTime;
            const absDx = Math.abs(dx);
            const absDy = Math.abs(dy);

            // Only trigger swipe if horizontal distance > 50px,
            // horizontal > vertical, and duration < 500ms
            if (absDx > 50 && absDx > absDy * 1.5 && dt < 500) {
                if (dx < 0) {
                    this.showNextImage();
                } else {
                    this.showPrevImage();
                }
            }
        }, { passive: true });
    }
    
    setupInfiniteScroll() {
        const options = {
            root: null,
            rootMargin: '100px',
            threshold: 0.1
        };
        
        this.observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting && !this.isLoading && this.hasMore) {
                this.loadMore();
            }
        }, options);
        
        if (this.scrollSentinel) {
            this.observer.observe(this.scrollSentinel);
        }
    }

    openModal(index) {
        if (index < 0 || index >= this.allImages.length) return;
        
        this.currentImageIndex = index;
        const imageData = this.allImages[index];
        const imageUrl = `/output/ranked/${encodeURIComponent(imageData.filename)}`;

        this.modalImage.src = imageUrl;
        this.modalFilename.textContent = imageData.filename;

        this.modalMetadata.innerHTML = `
            <div class="bg-black/30 px-3 py-1.5 rounded text-white"><span class="text-gray-400 mr-1">Score:</span> ${Utils.formatScore(imageData.score)}</div>
            <div class="bg-black/30 px-3 py-1.5 rounded text-white"><span class="text-gray-400 mr-1">Conf:</span> ${Utils.formatScore(imageData.confidence)}</div>
            <div class="bg-black/30 px-3 py-1.5 rounded text-white"><span class="text-gray-400 mr-1">Tier:</span> ${imageData.tier}</div>
            <div class="bg-black/30 px-3 py-1.5 rounded text-white"><span class="text-gray-400 mr-1">Comps:</span> ${imageData.comparison_count || 0}</div>
        `;

        this.modal.classList.remove("hidden");
        this.modal.classList.add("flex");
        
        // Hide/show arrows based on index
        this.modalPrev.style.display = index > 0 ? "block" : "none";
        this.modalNext.style.display = index < this.allImages.length - 1 ? "block" : "none";
        
        // Preload next/prev
        if (index < this.allImages.length - 1) {
            Utils.preloadImage(`/output/ranked/${encodeURIComponent(this.allImages[index+1].filename)}`);
        }
        if (index > 0) {
            Utils.preloadImage(`/output/ranked/${encodeURIComponent(this.allImages[index-1].filename)}`);
        }
        
        // If we reached near the end of loaded images while swiping right, fetch more in background
        if (index >= this.allImages.length - 5 && !this.isLoading && this.hasMore) {
            this.loadMore(false);
        }

        // Fetch and show history
        this.loadHistory(imageData.filename);
    }

    async loadHistory(filename) {
        try {
            this.modalHistory.classList.add("hidden");
            this.historyWins.innerHTML = "";
            this.historyLosses.innerHTML = "";

            const data = await api.getImageHistory(filename);
            if (!data.history || data.history.length === 0) return;

            this.modalHistory.classList.remove("hidden");
            
            data.history.forEach(item => {
                const isWin = item.winner;
                const isIndirect = item.indirect || false;
                const container = isWin ? this.historyWins : this.historyLosses;
                
                const el = document.createElement("div");
                el.className = "flex items-center gap-3 bg-white/5 border border-white/5 rounded-lg p-2 text-[11px] group hover:bg-white/10 transition-colors";
                
                const indirectBadge = isIndirect 
                    ? `<span class="bg-purple-500/20 text-purple-400 border border-purple-500/30 px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ml-2">Indirect</span>`
                    : "";

                const imgUrl = `/output/ranked/${encodeURIComponent(item.other)}`;
                const weight = item.weight ? `Weight: ${item.weight.toFixed(2)}` : "";
                const depth = item.transitive_depth > 0 ? `Inference (D${item.transitive_depth})` : "Direct";
                const depthClass = item.transitive_depth > 0 ? "text-purple-400" : "text-green-400";
                const dateShort = item.timestamp ? item.timestamp.split('T')[0] : "";

                el.innerHTML = `
                    <div class="w-10 h-10 rounded border border-white/10 overflow-hidden flex-shrink-0 bg-black/40">
                        <img src="${imgUrl}" alt="" class="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity">
                    </div>
                    <div class="flex flex-col overflow-hidden flex-grow text-left">
                        <span class="text-gray-300 truncate font-mono text-[10px]" title="${item.other || 'Unknown'}">${item.other || 'Unknown'}</span>
                        <div class="flex gap-2 text-[8px] font-bold uppercase tracking-tighter">
                            <span class="${depthClass}">${depth}</span>
                            <span class="text-gray-500">${weight}</span>
                        </div>
                    </div>
                    <div class="flex items-center flex-shrink-0 pr-1">
                        <span class="text-gray-600 text-[8px] font-mono">${dateShort}</span>
                    </div>
                `;
                container.appendChild(el);
            });

            if (this.historyWins.children.length === 0) {
                this.historyWins.innerHTML = '<div class="text-gray-500 italic text-center py-2">No wins recorded</div>';
            }
            if (this.historyLosses.children.length === 0) {
                this.historyLosses.innerHTML = '<div class="text-gray-500 italic text-center py-2">No losses recorded</div>';
            }

        } catch (error) {
            console.error("Failed to load history:", error);
        }
    }

    showNextImage() {
        if (this.currentImageIndex < this.allImages.length - 1) {
            this.openModal(this.currentImageIndex + 1);
        }
    }
    
    showPrevImage() {
        if (this.currentImageIndex > 0) {
            this.openModal(this.currentImageIndex - 1);
        }
    }

    closeModal() {
        this.modal.classList.add("hidden");
        this.modal.classList.remove("flex");
        this.modalImage.src = "";
    }

    async initialize() {
        this.updateStatus("Loading gallery...");
        await this.search();
    }

    async search() {
        this.currentPage = 1;
        this.allImages = [];
        this.galleryGrid.innerHTML = "";
        this.hasMore = true;
        await this.fetchAndRender();
    }
    
    async loadMore(renderStatus = true) {
        if (this.isLoading || !this.hasMore) return;
        this.currentPage++;
        await this.fetchAndRender(renderStatus);
    }

    async fetchAndRender(renderStatus = true) {
        try {
            this.isLoading = true;
            if (renderStatus) this.scrollSentinel.classList.remove("opacity-0");

            const filters = {
                scoreMin: parseFloat(this.scoreMin.value),
                scoreMax: parseFloat(this.scoreMax.value),
                confidenceMin: parseFloat(this.confidenceMin.value),
                confidenceMax: parseFloat(this.confidenceMax.value),
                comparisonsMin: parseInt(this.comparisonsMin.value),
                comparisonsMax: parseInt(this.comparisonsMax.value),
                sort: this.sortFilter.value,
            };

            const result = await api.getGalleryImages(this.currentPage, this.perPage, filters);
            if (this.currentPage * this.perPage >= result.total) {
                this.hasMore = false;
            }

            if (result.images && result.images.length > 0) {
                const startIndex = this.allImages.length;
                this.allImages = [...this.allImages, ...result.images];
                this.appendImages(result.images, startIndex);
                
                if (renderStatus) {
                    this.galleryStatus.style.display = "none";
                }
            } else if (this.allImages.length === 0) {
                this.galleryStatus.style.display = "block";
                this.updateStatus("No images found matching criteria.");
            }

            if (renderStatus) {
                this.statusBadge.textContent = `${this.allImages.length} images`;
                this.statusBadge.className = "px-3 py-1 text-sm font-semibold rounded-full bg-green-500/20 text-green-400 border border-green-500/30";
                
                if (!this.hasMore && this.allImages.length > 0) {
                    this.scrollSentinel.innerHTML = "End of results";
                    this.scrollSentinel.classList.remove("animate-pulse");
                }
            }
        } catch (error) {
            console.error(error);
            if (renderStatus) this.updateStatus(`Error: ${error.message}`);
            Utils.showToast(`Error: ${error.message}`, "error");
        } finally {
            this.isLoading = false;
            if (renderStatus && this.hasMore) this.scrollSentinel.classList.remove("opacity-0");
            else if (!this.hasMore) this.scrollSentinel.classList.remove("opacity-0");
        }
    }

    appendImages(images, startIndex) {
        const fragment = document.createDocumentFragment();
        
        images.forEach((image, i) => {
            const index = startIndex + i;
            const item = document.createElement("div");
            item.className = "gallery-item bg-dark-800 rounded-xl overflow-hidden cursor-pointer border border-white/5 transition-all duration-300 relative group";
            
            if (image.tier === undefined) image.tier = Math.floor(image.score * 10);
            
            item.innerHTML = `
                <div class="aspect-square w-full overflow-hidden bg-black/50">
                    <img src="/output/ranked/${encodeURIComponent(image.filename)}" alt="Img" loading="lazy" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500">
                </div>
                <div class="p-3 bg-dark-800/90 absolute bottom-0 w-full transform translate-y-full group-hover:translate-y-0 transition-transform duration-300 border-t border-white/10 backdrop-blur-md">
                    <div class="grid grid-cols-2 gap-1 text-[10px] text-gray-300 font-medium">
                        <div class="bg-black/40 px-2 py-1 rounded flex justify-between"><span>Score</span><span class="text-white">${Utils.formatScore(image.score)}</span></div>
                        <div class="bg-black/40 px-2 py-1 rounded flex justify-between"><span>Conf</span><span class="text-white">${Utils.formatScore(image.confidence)}</span></div>
                        <div class="bg-black/40 px-2 py-1 rounded flex justify-between"><span>Tier</span><span class="text-white">${image.tier}</span></div>
                        <div class="bg-black/40 px-2 py-1 rounded flex justify-between text-purple-300"><span>Comps</span><span class="text-white">${image.comparison_count || 0}</span></div>
                    </div>
                </div>
                <div class="absolute top-2 right-2 bg-black/60 backdrop-blur border border-white/10 text-white text-[10px] px-2 py-1 rounded-md font-bold shadow group-hover:opacity-0 transition-opacity">
                    ${Utils.formatScore(image.score)}
                </div>
            `;

            item.addEventListener("click", () => this.openModal(index));
            fragment.appendChild(item);
        });
        
        this.galleryGrid.appendChild(fragment);
    }

    updateStatus(message) {
        this.galleryStatus.textContent = message;
        this.galleryStatus.style.display = "block";
    }
}

document.addEventListener("DOMContentLoaded", () => {
    window.gallery = new GalleryApp();
});
