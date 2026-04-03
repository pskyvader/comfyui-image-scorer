/**
 * Gallery Mode JavaScript
 * Manages the scored images gallery view with filtering and infinite scroll capabilities
 */

// ==========================================
// STATE & CONFIGURATION
// ==========================================
const ITEMS_PER_PAGE = 20;

let galleryData = [];
let filteredData = [];
let currentPage = 1;
let totalGalleryCount = 0;
let currentImageIndexInModal = -1;

let imageObserver = null;
let scrollObserver = null;

let isLoadingMore = false;
let infiniteScrollEnabled = true;

let touchStartX = 0;
let touchEndX = 0;

let filterState = {
    effectiveScoreMin: 1,
    effectiveScoreMax: 5,
    comparisonsMin: 0,
    comparisonsMax: 10,
    volatilityMin: 0,
    volatilityMax: 1,
};

// ==========================================
// INITIALIZATION
// ==========================================

async function initializeGallery() {
    console.log("Initializing gallery view...");
    
    initializeLazyLoadObserver();
    loadFilterState();
    attachFilterListeners();
    setupSwipeGestures();
    
    // Setup infinite scroll observer immediately so it's ready
    setupInfiniteScroll();

    // Small delay to ensure DOM is fully ready before fetching
    await new Promise(resolve => setTimeout(resolve, 100));
    await loadGalleryPage(1);
}

function initializeLazyLoadObserver() {
    if (imageObserver) imageObserver.disconnect();
    
    imageObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                const img = entry.target;
                const fullSrc = img.dataset.src;
                if (fullSrc && !img.src.includes('/image/')) {
                    img.src = fullSrc;
                    imageObserver.unobserve(img);
                }
            }
        });
    }, { rootMargin: '50px' });
}

// ==========================================
// DATA LOADING & RENDERING
// ==========================================

async function loadGalleryData() {
    await loadGalleryPage(1);
}

async function loadGalleryPage(page = 1) {
    if (isLoadingMore && page !== 1) return;
    isLoadingMore = true;

    const statusText = document.getElementById("status-text");
    if (statusText && page === 1) statusText.innerText = "Loading gallery images...";

    try {
        const params = new URLSearchParams({
            page: page,
            per_page: ITEMS_PER_PAGE,
            effective_score_min: filterState.effectiveScoreMin,
            effective_score_max: filterState.effectiveScoreMax,
            comparisons_min: filterState.comparisonsMin,
            comparisons_max: filterState.comparisonsMax,
            volatility_min: filterState.volatilityMin,
            volatility_max: filterState.volatilityMax,
        });

        const res = await fetch(`/api/scores?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        
        const data = await res.json();
        const newItems = data.scores || [];
        totalGalleryCount = data.total || 0;
        currentPage = data.page || page;

        if (page === 1) {
            galleryData = newItems;
            filteredData = [...newItems];
            renderGalleryGrid();
        } else {
            galleryData = [...galleryData, ...newItems];
            filteredData = [...galleryData];
            appendGalleryItems(newItems);
        }

        updatePagination();

    } catch (error) {
        console.error("Error loading gallery page:", error);
        const grid = document.getElementById("gallery-grid");
        if (grid && currentPage === 1) {
            grid.innerHTML = '<div class="gallery-empty">Error loading images</div>';
        }
        if (statusText) statusText.innerText = "Error loading gallery.";
    } finally {
        // THE FIX: Use a debounce to yield to the browser's rendering engine.
        // This gives the DOM time to paint the new images, expand the container height,
        // and physically push the sentinel down before we check if we need to load more.
        setTimeout(() => {
            isLoadingMore = false;
            
            // Safely check if the screen is still empty after the DOM has updated
            const sentinel = document.querySelector('.gallery-sentinel');
            if (sentinel) {
                const rect = sentinel.getBoundingClientRect();
                // If sentinel is still visible, fetch the next page
                if (rect.top < window.innerHeight && currentPage * ITEMS_PER_PAGE < totalGalleryCount) {
                    loadGalleryPage(currentPage + 1);
                }
            }
        }, 250); // 250ms debounce prevents rapid-fire API spam
    }
}

function renderGalleryGrid() {
    const grid = document.getElementById("gallery-grid");
    if (!grid) return;
    
    grid.innerHTML = ""; // This destroys inner content

    if (!filteredData.length) {
        grid.innerHTML = '<div class="gallery-empty">No images match the selected filters</div>';
        return;
    }

    appendGalleryItems(filteredData);
}

function appendGalleryItems(items) {
    const grid = document.getElementById('gallery-grid');
    if (!grid) return;

    const frag = document.createDocumentFragment();
    items.forEach(item => {
        frag.appendChild(createGalleryItem(item));
    });

    grid.appendChild(frag);
}

function createGalleryItem(item) {
    const div = document.createElement("div");
    div.className = "gallery-item";
    
    const score = (Number(item.score) || 0).toFixed(2);
    const modifier = (Number(item.modifier) || 0).toFixed(2);
    const comparisons = parseInt(item.comparison_count) || 0;
    const volatility = Math.abs(Number(item.volatility) || 0).toFixed(2);
    const effectiveScore = (Number(score) + Number(modifier) / 10).toFixed(2);

    const imgSrc = `/image/${encodeURI(item.file_id)}`;
    const thumbSrc = `/thumbnail/${encodeURI(item.file_id)}`;

    div.innerHTML = `
        <div class="gallery-item-image">
            <img class="gallery-image" 
                 src="${thumbSrc}" 
                 data-src="${imgSrc}"
                 alt="${item.file_id}" 
                 onerror="this.src='${imgSrc}'; this.parentElement.title='Thumbnail failed, loaded full image'">
        </div>
        <div class="gallery-item-info">
            <div class="gallery-item-name">${item.file_id}</div>
            <div class="gallery-item-score">Effective Score: <strong>${effectiveScore}</strong></div>
            <div class="gallery-item-meta">
                <span>Base: <strong>${score}</strong> | Mod: <strong>${modifier > 0 ? '+' : ''}${modifier}</strong></span>
                <span>Comparisons: <strong>${comparisons}</strong></span>
                <span>Volatility: <strong>${volatility}</strong></span>
            </div>
        </div>
    `;

    const img = div.querySelector('.gallery-image');
    if (imageObserver && img) imageObserver.observe(img);

    div.addEventListener("click", () => showImageDetails(item));
    return div;
}

// ==========================================
// INFINITE SCROLL
// ==========================================

function setupInfiniteScroll() {
    const grid = document.getElementById('gallery-grid');
    if (!grid) return;

    // Disconnect existing observer if re-initializing
    if (scrollObserver) scrollObserver.disconnect();

    // Create sentinel OUTSIDE the grid, so grid.innerHTML = "" doesn't destroy it.
    let sentinel = document.querySelector('.gallery-sentinel');
    if (!sentinel) {
        sentinel = document.createElement('div');
        sentinel.className = 'gallery-sentinel';
        sentinel.style.height = '1px';
        sentinel.style.visibility = 'hidden';
        // Append as a sibling of the grid
        grid.parentNode.insertBefore(sentinel, grid.nextSibling);
    }

    scrollObserver = new IntersectionObserver(
        async (entries) => {
            if (entries[0].isIntersecting && !isLoadingMore && infiniteScrollEnabled) {
                const totalPages = Math.max(1, Math.ceil(totalGalleryCount / ITEMS_PER_PAGE));
                if (currentPage < totalPages) {
                    await loadGalleryPage(currentPage + 1);
                }
            }
        },
        { root: null, rootMargin: '500px' }
    );

    scrollObserver.observe(sentinel);
}

// ==========================================
// MODAL & NAVIGATION
// ==========================================

function showImageDetails(item) {
    currentImageIndexInModal = galleryData.findIndex(img => img.file_id === item.file_id);
    const modal = document.getElementById('image-modal');
    if (!modal) return;
    
    const modalImg = document.getElementById('modal-image');
    const modalTitle = document.getElementById('modal-title');
    
    if (modalImg) {
        modalImg.src = `/image/${encodeURI(item.file_id)}`;
        modalImg.alt = item.file_id;
    }
    if (modalTitle) modalTitle.textContent = item.file_id;
    
    updateModalNavigation();
    modal.style.display = 'block';
}

function previousImageInModal() {
    if (currentImageIndexInModal > 0) {
        currentImageIndexInModal--;
        showImageDetails(galleryData[currentImageIndexInModal]);
    }
}

async function nextImageInModal() {
    if (currentImageIndexInModal < galleryData.length - 1) {
        currentImageIndexInModal++;
        showImageDetails(galleryData[currentImageIndexInModal]);
    } else {
        const totalPages = Math.max(1, Math.ceil(totalGalleryCount / ITEMS_PER_PAGE));
        if (currentPage < totalPages && !isLoadingMore) {
            await loadGalleryPage(currentPage + 1);
            if (currentImageIndexInModal < galleryData.length - 1) {
                currentImageIndexInModal++;
                showImageDetails(galleryData[currentImageIndexInModal]);
            }
        }
    }
}

function updateModalNavigation() {
    const prevBtn = document.getElementById('modal-prev-btn');
    const nextBtn = document.getElementById('modal-next-btn');
    const totalPages = Math.max(1, Math.ceil(totalGalleryCount / ITEMS_PER_PAGE));

    if (prevBtn) prevBtn.disabled = currentImageIndexInModal <= 0;
    if (nextBtn) {
        const isAtEnd = currentImageIndexInModal >= galleryData.length - 1 && currentPage >= totalPages;
        nextBtn.disabled = isAtEnd;
    }
}

function closeImageModal() {
    const modal = document.getElementById('image-modal');
    if (modal) modal.style.display = 'none';
}

// ==========================================
// SWIPE GESTURES
// ==========================================

function setupSwipeGestures() {
    const modal = document.getElementById('image-modal');
    if (!modal) return;

    modal.addEventListener('touchstart', (e) => {
        if (e.target.closest('.modal-image')) {
            touchStartX = e.changedTouches[0].screenX;
        }
    }, { passive: true });

    modal.addEventListener('touchend', (e) => {
        if (e.target.closest('.modal-image')) {
            touchEndX = e.changedTouches[0].screenX;
            handleSwipeGesture();
        }
    }, { passive: true });
}

function handleSwipeGesture() {
    const swipeThreshold = 50; 
    const diff = touchStartX - touchEndX;

    if (Math.abs(diff) >= swipeThreshold) {
        if (diff > 0) nextImageInModal(); // Swiped left
        else previousImageInModal();      // Swiped right
    }
}

// ==========================================
// FILTERS
// ==========================================

function attachFilterListeners() {
    const bindFilter = (inputId, displayId, stateKey, isInt = false) => {
        const input = document.getElementById(inputId);
        if (input) {
            input.addEventListener("input", async (e) => {
                const val = isInt ? parseInt(e.target.value) : parseFloat(e.target.value);
                filterState[stateKey] = val;
                const display = document.getElementById(displayId);
                if (display) display.textContent = e.target.value;
                await new Promise(resolve => setTimeout(resolve, 500)); // Yield to ensure UI updates before processing
                applyFilters();
            });
        }
    };

    bindFilter("filter-effective-score-min", "effective-score-min-val", "effectiveScoreMin");
    bindFilter("filter-effective-score-max", "effective-score-max-val", "effectiveScoreMax");
    bindFilter("filter-comparisons-min", "comparisons-min-val", "comparisonsMin", true);
    bindFilter("filter-comparisons-max", "comparisons-max-val", "comparisonsMax", true);
    bindFilter("filter-volatility-min", "volatility-min-val", "volatilityMin");
    bindFilter("filter-volatility-max", "volatility-max-val", "volatilityMax");
}

function applyFilters() {
    currentPage = 1;
    loadGalleryPage(1);
    saveFilterState();
}

function resetFilters() {
    filterState = {
        effectiveScoreMin: 1, effectiveScoreMax: 5,
        comparisonsMin: 0, comparisonsMax: 10,
        volatilityMin: 0, volatilityMax: 1,
    };

    const updates = [
        { id: "filter-effective-score-min", val: 1, textId: "effective-score-min-val", text: "1.0" },
        { id: "filter-effective-score-max", val: 5, textId: "effective-score-max-val", text: "5.0" },
        { id: "filter-comparisons-min", val: 0, textId: "comparisons-min-val", text: "0" },
        { id: "filter-comparisons-max", val: 10, textId: "comparisons-max-val", text: "10" },
        { id: "filter-volatility-min", val: 0, textId: "volatility-min-val", text: "0.0" },
        { id: "filter-volatility-max", val: 1, textId: "volatility-max-val", text: "1.0" }
    ];

    updates.forEach(u => {
        const el = document.getElementById(u.id);
        const textEl = document.getElementById(u.textId);
        if (el) el.value = u.val;
        if (textEl) textEl.textContent = u.text;
    });

    applyFilters();
}

function toggleAdvancedFilters() {
    const advancedFilters = document.getElementById('advanced-filters');
    const toggleBtn = document.getElementById('toggle-filters-btn');
    if (advancedFilters) {
        const isHidden = advancedFilters.style.display === 'none';
        advancedFilters.style.display = isHidden ? 'block' : 'none';
        if (toggleBtn) {
            toggleBtn.textContent = isHidden ? 'Hide Advanced Filters' : 'Show Advanced Filters';
        }
    }
}

function saveFilterState() {
    try {
        localStorage.setItem("galleryFilterState", JSON.stringify(filterState));
    } catch (error) {
        console.warn("Could not save filter state:", error);
    }
}

function loadFilterState() {
    try {
        const saved = localStorage.getItem("galleryFilterState");
        if (saved) {
            filterState = { ...filterState, ...JSON.parse(saved) };
        }
    } catch (error) {
        console.warn("Could not load filter state:", error);
    }
}

// ==========================================
// UTILS & CLEANUP
// ==========================================

function updatePagination() {
    const statusText = document.getElementById("status-text");
    const totalPages = Math.max(1, Math.ceil(totalGalleryCount / ITEMS_PER_PAGE));
    
    if (statusText) {
        statusText.innerText = `Page ${currentPage} of ${totalPages} | Total: ${totalGalleryCount} images | Loaded: ${galleryData.length}`;
    }
}

function showError(message) {
    const grid = document.getElementById("gallery-grid");
    if (grid) {
        grid.innerHTML = `<div class="gallery-empty">${message}</div>`;
    }
}

function clearGalleryModeState() {
    galleryData = [];
    filteredData = [];
    currentPage = 1;
    
    closeImageModal();
    
    if (scrollObserver) {
        scrollObserver.disconnect();
        scrollObserver = null;
    }
    
    filterState = {
        effectiveScoreMin: 1, effectiveScoreMax: 5,
        comparisonsMin: 0, comparisonsMax: 10,
        volatilityMin: 0, volatilityMax: 1,
    };
}