// Service Worker for Sudoku Solver PWA
const CACHE_NAME = 'sudoku-solver-v1';
const STATIC_ASSETS = [
    '/',
    '/static/css/main.css',
    '/static/css/components.css',
    '/static/css/screens.css',
    '/static/css/animations.css',
    '/static/js/utils.js',
    '/static/js/api.js',
    '/static/js/history.js',
    '/static/js/grid.js',
    '/static/js/camera.js',
    '/static/js/screens.js',
    '/static/js/app.js',
    '/static/icons/icon-192.png',
    '/static/icons/icon-512.png',
    '/static/manifest.json'
];

// Install event - cache static assets
self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(STATIC_ASSETS))
            .then(() => self.skipWaiting())
    );
});

// Activate event - clean old caches
self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys =>
            Promise.all(
                keys.filter(key => key !== CACHE_NAME)
                    .map(key => caches.delete(key))
            )
        ).then(() => self.clients.claim())
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', event => {
    const { request } = event;
    const url = new URL(request.url);

    // API requests - network only
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(fetch(request));
        return;
    }

    // Static assets - cache first, fallback to network
    event.respondWith(
        caches.match(request).then(cached => {
            if (cached) {
                return cached;
            }

            return fetch(request).then(response => {
                // Cache successful responses for static assets
                if (response.ok && (url.pathname.startsWith('/static/') || url.pathname === '/')) {
                    const clone = response.clone();
                    caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
                }
                return response;
            });
        }).catch(() => {
            // Offline fallback for navigation
            if (request.mode === 'navigate') {
                return caches.match('/');
            }
        })
    );
});
