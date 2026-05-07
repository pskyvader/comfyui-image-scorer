/**
 * Gallery View Logic (SPA Compatible)
 */

class GalleryView {
    constructor() {
        this.currentPage = 1;
        this.perPage = 48;
        this.allImages = [];
        this.currentImageIndex = -1;
        this.isLoading = false;
        this.hasMore = true;
        this.els = {};
    }

    cacheElements() {
        this.els = {
            scoreMin: document.getElementById("score-min"),
            scoreMax: document.getElementById("score-max"),
            comparisonsMin: document.getElementById("comparisons-min"),
            comparisonsMax: document.getElementById("comparisons-max"),
            scoreDisplay: document.getElementById("score-display"),
            comparisonsDisplay: document.getElementById("comparisons-display"),
            sortFilter: document.getElementById("sort-filter"),
            galleryStatus: document.getElementById("gallery-status"),
            galleryGrid: document.getElementById("gallery-grid"),
            scrollSentinel: document.getElementById("scroll-sentinel"),
            modal: document.getElementById("image-modal"),
            modalImage: document.getElementById("modal-image"),
            modalFilename: document.getElementById("modal-filename"),
            modalMetadata: document.getElementById("modal-metadata"),
            modalClose: document.querySelector(".modal-close"),
            modalPrev: document.getElementById("modal-prev"),
            modalNext: document.getElementById("modal-next"),
            modalHistory: document.getElementById("modal-history"),
            historyWins: document.getElementById("history-wins"),
            historyLosses: document.getElementById("history-losses")
        };
    }

    attachEventListeners() {
        const updateRanges = () => {
            if (this.els.scoreMin && this.els.scoreMax) {
                if (parseFloat(this.els.scoreMin.value) > parseFloat(this.els.scoreMax.value)) {
                    this.els.scoreMin.value = this.els.scoreMax.value;
                }
                this.els.scoreDisplay.textContent = `${parseFloat(this.els.scoreMin.value).toFixed(2)} - ${parseFloat(this.els.scoreMax.value).toFixed(2)}`;
            }
            if (this.els.comparisonsMin && this.els.comparisonsMax) {
                if (parseInt(this.els.comparisonsMin.value) > parseInt(this.els.comparisonsMax.value)) {
                    this.els.comparisonsMin.value = this.els.comparisonsMax.value;
                }
                this.els.comparisonsDisplay.textContent = `${this.els.comparisonsMin.value} - ${this.els.comparisonsMax.value}`;
            }
        };

        const debouncedSearch = Utils.debounce(() => this.search(), 300);
        
        ['scoreMin', 'scoreMax', 'comparisonsMin', 'comparisonsMax'].forEach(id => {
            this.els[id]?.addEventListener("input", () => { updateRanges(); debouncedSearch(); });
        });
        
        this.els.sortFilter?.addEventListener("change", () => this.search());
        
        if (this.els.modalClose) this.els.modalClose.onclick = () => this.closeModal();
        if (this.els.modalPrev) this.els.modalPrev.onclick = (e) => { e.stopPropagation(); this.showPrevImage(); };
        if (this.els.modalNext) this.els.modalNext.onclick = (e) => { e.stopPropagation(); this.showNextImage(); };
        
        // Close modal on click outside
        this.els.modal?.addEventListener("click", (e) => {
            if (e.target === this.els.modal) this.closeModal();
        });

        // Key bindings
        document.addEventListener("keydown", (e) => {
            if (this.els.modal && !this.els.modal.classList.contains("hidden")) {
                if (e.key === "ArrowLeft") this.showPrevImage();
                if (e.key === "ArrowRight") this.showNextImage();
                if (e.key === "Escape") this.closeModal();
            }
        });

        // Swipe support
        let touchStartX = 0;
        this.els.modal?.addEventListener("touchstart", (e) => {
            touchStartX = e.changedTouches[0].screenX;
        }, { passive: true });
        this.els.modal?.addEventListener("touchend", (e) => {
            const touchEndX = e.changedTouches[0].screenX;
            if (touchStartX - touchEndX > 50) this.showNextImage();
            if (touchEndX - touchStartX > 50) this.showPrevImage();
        }, { passive: true });
    }

    async init() {
        console.log("Initializing GalleryView...");
        this.cacheElements();
        this.attachEventListeners();
        this.setupInfiniteScroll();
        await this.search();
    }

    setupInfiniteScroll() {
        if (this.observer) this.observer.disconnect();
        
        this.observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting && !this.isLoading && this.hasMore) {
                this.loadMore();
            }
        }, { rootMargin: '200px' });
        
        if (this.els.scrollSentinel) this.observer.observe(this.els.scrollSentinel);
    }

    async search() {
        this.currentPage = 1;
        this.allImages = [];
        this.els.galleryGrid.innerHTML = "";
        this.hasMore = true;
        await this.fetchAndRender();
    }

    async loadMore() {
        if (this.isLoading || !this.hasMore) return;
        this.currentPage++;
        await this.fetchAndRender();
    }

    async fetchAndRender() {
        try {
            this.isLoading = true;
            this.els.scrollSentinel?.classList.remove("opacity-0");

            const filters = {
                scoreMin: parseFloat(this.els.scoreMin.value),
                scoreMax: parseFloat(this.els.scoreMax.value),
                comparisonsMin: parseInt(this.els.comparisonsMin.value),
                comparisonsMax: parseInt(this.els.comparisonsMax.value),
                sort: this.els.sortFilter.value,
            };

            const result = await api.getGalleryImages(this.currentPage, this.perPage, filters);
            this.hasMore = (result.images?.length === this.perPage);

            if (result.images?.length > 0) {
                const startIndex = this.allImages.length;
                this.allImages = [...this.allImages, ...result.images];
                this.appendImages(result.images, startIndex);
                this.els.galleryStatus.classList.add("hidden");
                this.updateCount(result.total);
            } else if (this.allImages.length === 0) {
                this.els.galleryStatus.textContent = "No images found matching criteria.";
                this.els.galleryStatus.classList.remove("hidden");
                if (this.els.galleryCount) this.els.galleryCount.textContent = "0 images found";
            }
        } catch (e) {
            console.error("Gallery fetch failed:", e);
        } finally {
            this.isLoading = false;
            this.els.scrollSentinel?.classList.add("opacity-0");
        }
    }

    updateCount(total) {
        if (!this.els.galleryCount) {
            this.els.galleryCount = document.createElement("div");
            this.els.galleryCount.id = "gallery-count";
            this.els.galleryCount.className = "text-xs text-purple-400 font-bold mb-4 bg-purple-500/10 px-3 py-1 rounded-full border border-purple-500/20 inline-block";
            this.els.galleryGrid.parentNode.insertBefore(this.els.galleryCount, this.els.galleryGrid);
        }
        this.els.galleryCount.textContent = `Showing ${this.allImages.length} of ${total} images`;
    }

    appendImages(images, startIndex) {
        images.forEach((img, i) => {
            const index = startIndex + i;
            const item = document.createElement("div");
            item.className = "gallery-item bg-dark-800 rounded-xl overflow-hidden cursor-pointer border border-white/5 transition-all duration-300 relative group aspect-square";
            item.innerHTML = `
                <img src="/output/ranked/${encodeURIComponent(img.filename)}" loading="lazy" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500">
                <div class="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-3">
                    <div class="text-[10px] text-gray-300 truncate">${img.filename}</div>
                </div>
                <div class="absolute top-2 right-2 bg-black/60 backdrop-blur border border-white/10 text-white text-[10px] px-2 py-1 rounded-md font-bold shadow">
                    ${Utils.formatScore(img.score)}
                </div>
            `;
            item.onclick = () => this.openModal(index);
            this.els.galleryGrid.appendChild(item);
        });
    }

    async openModal(index) {
        this.currentImageIndex = index;
        const img = this.allImages[index];
        if (!img) return;

        this.els.modalImage.src = `/output/ranked/${encodeURIComponent(img.filename)}`;
        this.els.modalFilename.textContent = img.filename;
        
        this.els.modalMetadata.innerHTML = `
            <div class="bg-white/5 px-3 py-1.5 rounded-lg border border-white/5">
                <span class="text-gray-400 block uppercase text-[8px] tracking-widest mb-0.5">Score</span>
                <span class="text-white font-bold">${Utils.formatScore(img.score)}</span>
            </div>
            <div class="bg-white/5 px-3 py-1.5 rounded-lg border border-white/5">
                <span class="text-gray-400 block uppercase text-[8px] tracking-widest mb-0.5">Comparisons</span>
                <span class="text-white font-bold">${img.comparison_count ?? 0}</span>
            </div>
            <div class="bg-white/5 px-3 py-1.5 rounded-lg border border-white/5">
                <span class="text-gray-400 block uppercase text-[8px] tracking-widest mb-0.5">Chain</span>
                <span class="text-white font-bold">${img.chain_length ?? img.height ?? "-"}</span>
            </div>
        `;

        this.els.modal.classList.remove("hidden");
        this.els.modal.classList.add("flex");
        document.body.style.overflow = "hidden";

        this.loadHistory(img.filename);
    }

    closeModal() {
        this.els.modal.classList.add("hidden");
        this.els.modal.classList.remove("flex");
        document.body.style.overflow = "";
    }

    async loadHistory(filename) {
        this.els.modalHistory.classList.add("hidden");
        this.els.historyWins.innerHTML = "";
        this.els.historyLosses.innerHTML = "";

        try {
            const history = await api.getImageHistory(filename);
            if (!history || (!history.wins?.length && !history.losses?.length)) return;

            const renderItem = (item) => `
                <div class="flex items-center gap-3 p-2 bg-white/5 rounded-lg border border-white/5">
                    <img src="/images/${encodeURIComponent(item.opponent)}" class="w-10 h-10 rounded object-cover border border-white/10" onerror="this.src='/output/ranked/${encodeURIComponent(item.opponent)}'">
                    <div class="flex-1 min-w-0">
                        <div class="text-[10px] text-gray-300 truncate">${item.opponent || "Unknown"}</div>
                        <div class="text-[9px] text-purple-400 font-bold">${Utils.formatScore(item.opponent_score)}</div>
                    </div>
                </div>
            `;

            if (history.wins?.length) {
                this.els.historyWins.innerHTML = history.wins.map(renderItem).join("");
            } else {
                this.els.historyWins.innerHTML = '<div class="text-center py-4 text-gray-500 text-[10px]">No wins yet</div>';
            }

            if (history.losses?.length) {
                this.els.historyLosses.innerHTML = history.losses.map(renderItem).join("");
            } else {
                this.els.historyLosses.innerHTML = '<div class="text-center py-4 text-gray-500 text-[10px]">No losses yet</div>';
            }

            this.els.modalHistory.classList.remove("hidden");
        } catch (e) {
            console.error("History load failed:", e);
        }
    }
    
    showNextImage() {
        if (this.currentImageIndex < this.allImages.length - 1) {
            this.openModal(this.currentImageIndex + 1);
        } else if (this.hasMore) {
            this.loadMore().then(() => {
                if (this.currentImageIndex < this.allImages.length - 1) {
                    this.openModal(this.currentImageIndex + 1);
                }
            });
        }
    }

    showPrevImage() {
        if (this.currentImageIndex > 0) {
            this.openModal(this.currentImageIndex - 1);
        }
    }
}

window.galleryView = new GalleryView();
