/* Native cards remain native. State is removed immediately for other engines. */
(() => {
    let previousRefresh;
    let refreshOverridden = false;
    let booguSelected = false;
    function refreshBooguCards() {
        const root = typeof gradioApp === 'function' ? gradioApp() : document;
        const selector = root.querySelector('#setting_sd_model_checkpoint');
        const input = selector && selector.querySelector('input');
        const text = input ? input.value : '';
        const match = text.match(/Boogu\s*[—-]\s*(edit-turbo|turbo|edit|base)\s*[—-]/);
        const trunk = match ? match[1] : null;
        booguSelected = !!trunk;
        if (typeof opts !== 'undefined') {
            if (trunk) {
                if (!refreshOverridden) previousRefresh = opts.live_preview_refresh_period;
                opts.live_preview_refresh_period = 1000;
                refreshOverridden = true;
            } else if (refreshOverridden) {
                opts.live_preview_refresh_period = previousRefresh;
                refreshOverridden = false;
            }
        }
        root.querySelectorAll('.boogu-lora').forEach(card => {
            const unknown = card.classList.contains('boogu-target-unknown') || card.classList.contains('boogu-target-all');
            const foreign = !!trunk && ((!unknown && !card.classList.contains('boogu-target-' + trunk)) || card.classList.contains('boogu-distill'));
            card.classList.toggle('boogu-foreign', foreign);
            const message = card.classList.contains('boogu-target-foreign') ? 'not a Boogu LoRA' : 'incompatible Boogu trunk / merged distill';
            if (card.dataset.booguMessage !== message) card.dataset.booguMessage = message;
        });
    }
    document.addEventListener('click', event => {
        const card = event.target.closest && event.target.closest('.boogu-foreign');
        if (card && !event.target.closest('.button-row')) {
            event.preventDefault();
            event.stopImmediatePropagation();
        }
    }, true);
    if (typeof onUiUpdate === 'function') onUiUpdate(refreshBooguCards);
    if (typeof onUiLoaded === 'function') onUiLoaded(refreshBooguCards);
    if (typeof onUiLoaded === 'function') onUiLoaded(() => {
        if (typeof requestProgress !== 'function' || requestProgress.booguDual) return;
        const previous = requestProgress;
        const wrapped = function(id, container, gallery, atEnd, onProgress, ...rest) {
            if (!booguSelected) return previous.call(this, id, container, gallery, atEnd, onProgress, ...rest);
            const current = document.createElement('div');
            current.className = 'progressDiv boogu-current-progress';
            current.setAttribute('aria-label', 'Current image progress');
            const fill = document.createElement('div');
            fill.className = 'progress';
            fill.style.width = '0%';
            fill.textContent = 'Current image: 0%';
            current.appendChild(fill);
            container.parentNode.insertBefore(current, container);
            const end = (...args) => { current.remove(); if (atEnd) return atEnd(...args); };
            const progress = res => {
                const match = (res.textinfo || '').match(/\[Boogu steps (\d+)\/(\d+)\]/);
                const value = match ? Math.min(100, 100 * Number(match[1]) / Math.max(1, Number(match[2]))) : (/saving image/.test(res.textinfo || '') ? 100 : 0);
                fill.style.width = value + '%';
                fill.textContent = 'Current image: ' + Math.round(value) + '%';
                container.parentNode.querySelectorAll('.progressDiv:not(.boogu-current-progress) .progress').forEach(bar => {
                    bar.textContent = 'Overall: ' + Math.round((res.progress || 0) * 100) + '%';
                });
                if (onProgress) onProgress(res);
            };
            try { return previous.call(this, id, container, gallery, end, progress, ...rest); }
            catch (error) { current.remove(); throw error; }
        };
        wrapped.booguDual = true;
        requestProgress = wrapped;
    });
})();
