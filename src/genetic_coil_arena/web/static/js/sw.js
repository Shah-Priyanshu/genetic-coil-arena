// Service Worker for Genetic Coil Arena PWA
// Phase 4E: Mobile & Progressive Web App functionality

const CACHE_NAME = 'genetic-coil-arena-v1';
const STATIC_CACHE = 'static-cache-v1';
const DYNAMIC_CACHE = 'dynamic-cache-v1';

// Files to cache for offline functionality
const STATIC_FILES = [
    '/',
    '/static/css/main.css',
    '/static/js/main.js',
    '/static/js/mobile.js',
    '/static/js/pwa.js',
    '/static/images/icon-192.png',
    '/static/images/icon-512.png',
    '/manifest.json',
    '/offline.html'
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
    '/api/performance/status',
    '/api/performance/metrics',
    '/api/experiments/list',
    '/api/user/preferences'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    console.log('Service Worker: Installing...');
    
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then((cache) => {
                console.log('Service Worker: Caching static files');
                return cache.addAll(STATIC_FILES);
            })
            .catch((error) => {
                console.error('Service Worker: Failed to cache static files', error);
            })
    );
    
    // Activate immediately
    self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('Service Worker: Activating...');
    
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
                        console.log('Service Worker: Deleting old cache', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        })
    );
    
    // Take control of all pages
    self.clients.claim();
});

// Fetch event - network first strategy with fallback
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Handle different types of requests
    if (request.method === 'GET') {
        if (isStaticAsset(request.url)) {
            // Cache first strategy for static assets
            event.respondWith(cacheFirst(request));
        } else if (isAPIRequest(request.url)) {
            // Network first strategy for API requests
            event.respondWith(networkFirstAPI(request));
        } else {
            // Network first with offline fallback for pages
            event.respondWith(networkFirstPage(request));
        }
    }
});

// Cache first strategy for static assets
async function cacheFirst(request) {
    try {
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        const networkResponse = await fetch(request);
        const cache = await caches.open(STATIC_CACHE);
        cache.put(request, networkResponse.clone());
        return networkResponse;
    } catch (error) {
        console.error('Cache first strategy failed:', error);
        return new Response('Offline', { status: 503 });
    }
}

// Network first strategy for API requests
async function networkFirstAPI(request) {
    try {
        const networkResponse = await fetch(request);
        
        // Cache successful API responses
        if (networkResponse.status === 200) {
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.log('Network failed, trying cache for API request');
        const cachedResponse = await caches.match(request);
        
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Return offline API response
        return new Response(JSON.stringify({
            error: 'Offline',
            message: 'No network connection available',
            cached: false
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// Network first strategy for pages
async function networkFirstPage(request) {
    try {
        const networkResponse = await fetch(request);
        
        // Cache successful page responses
        if (networkResponse.status === 200) {
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.log('Network failed, trying cache for page');
        const cachedResponse = await caches.match(request);
        
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Return offline page
        return caches.match('/offline.html');
    }
}

// Helper functions
function isStaticAsset(url) {
    return url.includes('/static/') || 
           url.includes('manifest.json') ||
           url.includes('.css') ||
           url.includes('.js') ||
           url.includes('.png') ||
           url.includes('.jpg') ||
           url.includes('.ico');
}

function isAPIRequest(url) {
    return url.includes('/api/');
}

// Background sync for data synchronization
self.addEventListener('sync', (event) => {
    console.log('Service Worker: Background sync triggered', event.tag);
    
    if (event.tag === 'background-sync') {
        event.waitUntil(doBackgroundSync());
    }
});

async function doBackgroundSync() {
    try {
        // Sync cached experiment data
        const cache = await caches.open(DYNAMIC_CACHE);
        const cachedRequests = await cache.keys();
        
        for (const request of cachedRequests) {
            if (isAPIRequest(request.url)) {
                try {
                    const freshResponse = await fetch(request);
                    if (freshResponse.status === 200) {
                        await cache.put(request, freshResponse.clone());
                    }
                } catch (error) {
                    console.log('Background sync failed for:', request.url);
                }
            }
        }
        
        console.log('Background sync completed');
    } catch (error) {
        console.error('Background sync error:', error);
    }
}

// Push notification handling
self.addEventListener('push', (event) => {
    console.log('Service Worker: Push notification received');
    
    const options = {
        body: 'Your experiment has completed!',
        icon: '/static/images/icon-192.png',
        badge: '/static/images/badge-72.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            {
                action: 'explore',
                title: 'View Results',
                icon: '/static/images/checkmark.png'
            },
            {
                action: 'close',
                title: 'Close',
                icon: '/static/images/xmark.png'
            }
        ]
    };
    
    if (event.data) {
        const data = event.data.json();
        options.body = data.message || options.body;
        options.data = { ...options.data, ...data };
    }
    
    event.waitUntil(
        self.registration.showNotification('Genetic Coil Arena', options)
    );
});

// Notification click handling
self.addEventListener('notificationclick', (event) => {
    console.log('Service Worker: Notification clicked');
    
    event.notification.close();
    
    if (event.action === 'explore') {
        event.waitUntil(
            clients.openWindow('/experiments')
        );
    } else if (event.action === 'close') {
        // Just close the notification
        return;
    } else {
        // Default action - open the app
        event.waitUntil(
            clients.openWindow('/')
        );
    }
});

// Message handling from main thread
self.addEventListener('message', (event) => {
    console.log('Service Worker: Message received', event.data);
    
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
    
    if (event.data && event.data.type === 'CACHE_URLS') {
        event.waitUntil(
            caches.open(DYNAMIC_CACHE).then((cache) => {
                return cache.addAll(event.data.payload);
            })
        );
    }
});

console.log('Service Worker: Loaded and ready');
