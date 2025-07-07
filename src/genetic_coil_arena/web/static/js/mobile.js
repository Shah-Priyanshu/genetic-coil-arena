/**
 * Mobile & PWA JavaScript for Genetic Coil Arena
 * Phase 4E: Mobile optimization and progressive web app features
 */

class MobileManager {
    constructor() {
        this.deviceInfo = this.detectDevice();
        this.touchSupport = 'ontouchstart' in window;
        this.isInstalled = this.checkIfInstalled();
        this.networkStatus = navigator.onLine;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.initializeServiceWorker();
        this.setupInstallPrompt();
        this.optimizeForDevice();
        this.setupOfflineHandling();
        this.initializeNotifications();
        this.setupGestures();
        
        console.log('Mobile Manager initialized:', this.deviceInfo);
    }
    
    detectDevice() {
        const userAgent = navigator.userAgent;
        const screen = window.screen;
        
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(userAgent);
        const isTablet = /iPad|Android(?=.*\bMobile\b)(?=.*\bSafari\b)/i.test(userAgent) || 
                        (screen.width >= 768 && screen.width <= 1024);
        
        return {
            type: isMobile ? (isTablet ? 'tablet' : 'mobile') : 'desktop',
            width: screen.width,
            height: screen.height,
            pixelRatio: window.devicePixelRatio || 1,
            userAgent: userAgent,
            isMobile: isMobile,
            isTablet: isTablet,
            isTouch: this.touchSupport,
            orientation: screen.orientation ? screen.orientation.type : 'unknown'
        };
    }
    
    checkIfInstalled() {
        return window.matchMedia('(display-mode: standalone)').matches ||
               window.navigator.standalone ||
               document.referrer.includes('android-app://');
    }
    
    setupEventListeners() {
        // Network status
        window.addEventListener('online', () => {
            this.networkStatus = true;
            this.handleNetworkStatusChange(true);
        });
        
        window.addEventListener('offline', () => {
            this.networkStatus = false;
            this.handleNetworkStatusChange(false);
        });
        
        // Orientation change
        window.addEventListener('orientationchange', () => {
            setTimeout(() => this.handleOrientationChange(), 500);
        });
        
        // Viewport changes
        window.addEventListener('resize', this.debounce(() => {
            this.handleViewportChange();
        }, 250));
        
        // Touch events for mobile optimization
        if (this.touchSupport) {
            this.setupTouchEvents();
        }
        
        // Visibility change for performance optimization
        document.addEventListener('visibilitychange', () => {
            this.handleVisibilityChange();
        });
    }
    
    setupTouchEvents() {
        let touchStartY = 0;
        let touchEndY = 0;
        
        document.addEventListener('touchstart', (e) => {
            touchStartY = e.changedTouches[0].screenY;
        }, { passive: true });
        
        document.addEventListener('touchend', (e) => {
            touchEndY = e.changedTouches[0].screenY;
            this.handleSwipeGesture(touchStartY, touchEndY);
        }, { passive: true });
        
        // Prevent zoom on double tap for better UX
        let lastTouchEnd = 0;
        document.addEventListener('touchend', (e) => {
            const now = (new Date()).getTime();
            if (now - lastTouchEnd <= 300) {
                e.preventDefault();
            }
            lastTouchEnd = now;
        }, false);
    }
    
    async initializeServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register('/static/js/sw.js');
                console.log('Service Worker registered:', registration);
                
                // Listen for updates
                registration.addEventListener('updatefound', () => {
                    this.handleServiceWorkerUpdate(registration);
                });
                
                // Handle messages from service worker
                navigator.serviceWorker.addEventListener('message', (event) => {
                    this.handleServiceWorkerMessage(event);
                });
                
            } catch (error) {
                console.error('Service Worker registration failed:', error);
            }
        }
    }
    
    setupInstallPrompt() {
        let deferredPrompt;
        
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            this.showInstallButton(deferredPrompt);
        });
        
        window.addEventListener('appinstalled', () => {
            console.log('PWA installed successfully');
            this.hideInstallButton();
            this.trackEvent('pwa_installed');
        });
    }
    
    showInstallButton(deferredPrompt) {
        const installButton = document.createElement('button');
        installButton.className = 'install-button';
        installButton.textContent = '📱 Install App';
        installButton.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 25px;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            cursor: pointer;
            animation: slideIn 0.3s ease-out;
        `;
        
        installButton.addEventListener('click', async () => {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                const result = await deferredPrompt.userChoice;
                console.log('Install prompt result:', result);
                deferredPrompt = null;
                this.hideInstallButton();
            }
        });
        
        document.body.appendChild(installButton);
    }
    
    hideInstallButton() {
        const installButton = document.querySelector('.install-button');
        if (installButton) {
            installButton.remove();
        }
    }
    
    optimizeForDevice() {
        const body = document.body;
        
        // Add device classes
        body.classList.add(`device-${this.deviceInfo.type}`);
        if (this.deviceInfo.isTouch) body.classList.add('touch-device');
        if (this.isInstalled) body.classList.add('pwa-installed');
        
        // Mobile-specific optimizations
        if (this.deviceInfo.isMobile) {
            this.optimizeForMobile();
        }
        
        // High DPI optimizations
        if (this.deviceInfo.pixelRatio > 1) {
            body.classList.add('high-dpi');
        }
    }
    
    optimizeForMobile() {
        // Reduce animation complexity on mobile
        document.documentElement.style.setProperty('--animation-duration', '0.2s');
        
        // Optimize viewport for mobile
        let viewport = document.querySelector('meta[name="viewport"]');
        if (!viewport) {
            viewport = document.createElement('meta');
            viewport.name = 'viewport';
            document.head.appendChild(viewport);
        }
        viewport.content = 'width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes';
        
        // Add mobile-specific styles
        const mobileStyles = document.createElement('style');
        mobileStyles.textContent = `
            @media (max-width: 768px) {
                .desktop-only { display: none !important; }
                .mobile-hidden { display: none !important; }
                .container { padding: 10px; }
                .card { margin: 8px 0; }
                .btn { min-height: 44px; padding: 12px; }
                .table-responsive { overflow-x: auto; }
            }
        `;
        document.head.appendChild(mobileStyles);
    }
    
    setupOfflineHandling() {
        // Create offline indicator
        this.createOfflineIndicator();
        
        // Handle offline data storage
        this.setupOfflineStorage();
    }
    
    createOfflineIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'offline-indicator';
        indicator.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: #f39c12;
            color: white;
            text-align: center;
            padding: 10px;
            z-index: 10000;
            transform: translateY(-100%);
            transition: transform 0.3s ease;
        `;
        indicator.textContent = '📡 No internet connection - Working offline';
        document.body.appendChild(indicator);
    }
    
    setupOfflineStorage() {
        // Store user preferences locally
        this.preferences = this.loadPreferences();
        
        // Auto-save preferences on change
        window.addEventListener('beforeunload', () => {
            this.savePreferences();
        });
    }
    
    async initializeNotifications() {
        if ('Notification' in window && 'serviceWorker' in navigator) {
            const permission = await this.requestNotificationPermission();
            if (permission === 'granted') {
                this.setupPushNotifications();
            }
        }
    }
    
    async requestNotificationPermission() {
        if (Notification.permission === 'default') {
            return await Notification.requestPermission();
        }
        return Notification.permission;
    }
    
    async setupPushNotifications() {
        try {
            const registration = await navigator.serviceWorker.ready;
            const subscription = await registration.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: this.urlBase64ToUint8Array('YOUR_VAPID_PUBLIC_KEY')
            });
            
            // Send subscription to server
            await this.sendSubscriptionToServer(subscription);
        } catch (error) {
            console.error('Push notification setup failed:', error);
        }
    }
    
    setupGestures() {
        if (!this.touchSupport) return;
        
        let hammer;
        if (typeof Hammer !== 'undefined') {
            hammer = new Hammer(document.body);
            
            // Swipe gestures
            hammer.get('swipe').set({ direction: Hammer.DIRECTION_ALL });
            
            hammer.on('swipeleft', () => this.handleSwipe('left'));
            hammer.on('swiperight', () => this.handleSwipe('right'));
            hammer.on('swipeup', () => this.handleSwipe('up'));
            hammer.on('swipedown', () => this.handleSwipe('down'));
            
            // Pinch to zoom
            hammer.get('pinch').set({ enable: true });
            hammer.on('pinch', (e) => this.handlePinch(e));
        }
    }
    
    // Event Handlers
    handleNetworkStatusChange(isOnline) {
        const indicator = document.getElementById('offline-indicator');
        if (indicator) {
            indicator.style.transform = isOnline ? 'translateY(-100%)' : 'translateY(0)';
        }
        
        if (isOnline) {
            this.syncOfflineData();
        }
        
        this.trackEvent('network_status_change', { online: isOnline });
    }
    
    handleOrientationChange() {
        const orientation = screen.orientation ? screen.orientation.type : 
                          (window.innerWidth > window.innerHeight ? 'landscape' : 'portrait');
        
        document.body.classList.remove('portrait', 'landscape');
        document.body.classList.add(orientation.includes('portrait') ? 'portrait' : 'landscape');
        
        this.trackEvent('orientation_change', { orientation });
    }
    
    handleViewportChange() {
        this.deviceInfo.width = window.innerWidth;
        this.deviceInfo.height = window.innerHeight;
        
        // Update CSS custom properties
        document.documentElement.style.setProperty('--viewport-width', this.deviceInfo.width + 'px');
        document.documentElement.style.setProperty('--viewport-height', this.deviceInfo.height + 'px');
    }
    
    handleVisibilityChange() {
        const isHidden = document.hidden;
        
        if (isHidden) {
            // Pause expensive operations
            this.pauseAnimations();
        } else {
            // Resume operations
            this.resumeAnimations();
        }
        
        this.trackEvent('visibility_change', { hidden: isHidden });
    }
    
    handleSwipeGesture(startY, endY) {
        const threshold = 50;
        const diff = startY - endY;
        
        if (Math.abs(diff) > threshold) {
            if (diff > 0) {
                this.handleSwipe('up');
            } else {
                this.handleSwipe('down');
            }
        }
    }
    
    handleSwipe(direction) {
        console.log('Swipe detected:', direction);
        this.trackEvent('swipe_gesture', { direction });
        
        // Custom swipe handling based on current page
        const currentPage = window.location.pathname;
        
        if (currentPage.includes('/experiments/')) {
            if (direction === 'left') {
                this.navigateToNext();
            } else if (direction === 'right') {
                this.navigateToPrevious();
            }
        }
    }
    
    handlePinch(event) {
        const scale = event.scale;
        
        // Handle zoom on visualizations
        const visualizations = document.querySelectorAll('.visualization, .chart');
        visualizations.forEach(viz => {
            if (viz.contains(event.target)) {
                this.handleVisualizationZoom(viz, scale);
            }
        });
    }
    
    // Navigation Methods
    navigateToNext() {
        const nextButton = document.querySelector('.nav-next, .btn-next, [data-next]');
        if (nextButton) {
            nextButton.click();
            this.hapticFeedback();
        }
    }
    
    navigateToPrevious() {
        const prevButton = document.querySelector('.nav-prev, .btn-prev, [data-prev]');
        if (prevButton) {
            prevButton.click();
            this.hapticFeedback();
        }
    }
    
    // Visualization Methods
    handleVisualizationZoom(element, scale) {
        const currentScale = parseFloat(element.dataset.scale || 1);
        const newScale = Math.max(0.5, Math.min(3, currentScale * scale));
        
        element.style.transform = `scale(${newScale})`;
        element.dataset.scale = newScale;
        
        this.trackEvent('visualization_zoom', { scale: newScale });
    }
    
    // Animation Control
    pauseAnimations() {
        document.body.classList.add('animations-paused');
        
        // Pause CSS animations
        const style = document.createElement('style');
        style.id = 'pause-animations';
        style.textContent = `
            *, *::before, *::after {
                animation-play-state: paused !important;
                transition-duration: 0s !important;
            }
        `;
        document.head.appendChild(style);
    }
    
    resumeAnimations() {
        document.body.classList.remove('animations-paused');
        
        // Resume CSS animations
        const pauseStyle = document.getElementById('pause-animations');
        if (pauseStyle) {
            pauseStyle.remove();
        }
    }
    
    // Haptic Feedback
    hapticFeedback(type = 'light') {
        if ('vibrate' in navigator && this.preferences.vibration) {
            const patterns = {
                light: [10],
                medium: [20],
                heavy: [30],
                success: [10, 50, 10],
                error: [50, 100, 50]
            };
            
            navigator.vibrate(patterns[type] || patterns.light);
        }
    }
    
    // Service Worker Message Handling
    handleServiceWorkerUpdate(registration) {
        const newWorker = registration.installing;
        
        // Show update notification
        this.showUpdateNotification(() => {
            newWorker.postMessage({ action: 'SKIP_WAITING' });
            window.location.reload();
        });
    }
    
    handleServiceWorkerMessage(event) {
        const { type, data } = event.data;
        
        switch (type) {
            case 'BACKGROUND_SYNC_SUCCESS':
                this.showToast('Data synchronized successfully', 'success');
                break;
            case 'BACKGROUND_SYNC_FAILED':
                this.showToast('Sync failed - will retry later', 'warning');
                break;
            case 'CACHE_UPDATED':
                console.log('Cache updated with new content');
                break;
            case 'OFFLINE_FALLBACK':
                this.showToast('You are offline - showing cached content', 'info');
                break;
        }
    }
    
    // Notification Methods
    showUpdateNotification(onUpdate) {
        const notification = document.createElement('div');
        notification.className = 'update-notification';
        notification.innerHTML = `
            <div class="update-content">
                <span>🚀 New version available!</span>
                <button class="update-btn">Update</button>
                <button class="dismiss-btn">×</button>
            </div>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #4CAF50;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10001;
            animation: slideDown 0.3s ease-out;
        `;
        
        notification.querySelector('.update-btn').onclick = onUpdate;
        notification.querySelector('.dismiss-btn').onclick = () => notification.remove();
        
        document.body.appendChild(notification);
        
        // Auto-dismiss after 10 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 10000);
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        const colors = {
            success: '#4CAF50',
            warning: '#FF9800',
            error: '#F44336',
            info: '#2196F3'
        };
        
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: ${colors[type] || colors.info};
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            z-index: 10002;
            animation: fadeInUp 0.3s ease-out;
            max-width: 90vw;
            text-align: center;
        `;
        
        document.body.appendChild(toast);
        
        // Auto-remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'fadeOutDown 0.3s ease-out';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
    
    // Offline Data Management
    getOfflineData() {
        try {
            const data = localStorage.getItem('offline_data');
            return data ? JSON.parse(data) : [];
        } catch (error) {
            console.error('Error getting offline data:', error);
            return [];
        }
    }
    
    storeOfflineData(data) {
        try {
            const existing = this.getOfflineData();
            existing.push({
                ...data,
                timestamp: Date.now(),
                id: this.generateId()
            });
            localStorage.setItem('offline_data', JSON.stringify(existing));
        } catch (error) {
            console.error('Error storing offline data:', error);
        }
    }
    
    async uploadOfflineData(data) {
        try {
            const response = await fetch('/api/mobile/offline/upload-data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data })
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status}`);
            }
            
            this.showToast('Offline data synchronized', 'success');
            return true;
        } catch (error) {
            console.error('Failed to upload offline data:', error);
            this.showToast('Sync failed - will retry later', 'warning');
            return false;
        }
    }
    
    clearOfflineData() {
        localStorage.removeItem('offline_data');
    }
    
    // Push Notification Methods
    async sendSubscriptionToServer(subscription) {
        try {
            const response = await fetch('/api/mobile/push/subscribe', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(subscription)
            });
            
            if (response.ok) {
                console.log('Push subscription sent to server');
            }
        } catch (error) {
            console.error('Failed to send push subscription:', error);
        }
    }
    
    generateId() {
        return Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
    }
}

// Initialize mobile manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.mobileManager = new MobileManager();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MobileManager;
}

// CSS Animations for mobile interactions
const mobileAnimations = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideDown {
        from { transform: translateX(-50%) translateY(-100%); opacity: 0; }
        to { transform: translateX(-50%) translateY(0); opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from { transform: translateX(-50%) translateY(20px); opacity: 0; }
        to { transform: translateX(-50%) translateY(0); opacity: 1; }
    }
    
    @keyframes fadeOutDown {
        from { transform: translateX(-50%) translateY(0); opacity: 1; }
        to { transform: translateX(-50%) translateY(20px); opacity: 0; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    /* Mobile-specific styles */
    .touch-device .btn:active {
        transform: scale(0.95);
        transition: transform 0.1s ease;
    }
    
    .touch-device .card:active {
        transform: scale(0.98);
        transition: transform 0.1s ease;
    }
    
    /* PWA-specific styles */
    .pwa-installed .install-button {
        display: none !important;
    }
    
    /* Offline indicator styles */
    #offline-indicator {
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Update notification styles */
    .update-notification {
        font-size: 14px;
        font-weight: 500;
    }
    
    .update-notification .update-content {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .update-notification .update-btn {
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        padding: 6px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12px;
    }
    
    .update-notification .dismiss-btn {
        background: none;
        border: none;
        color: white;
        font-size: 16px;
        cursor: pointer;
        padding: 2px 6px;
        border-radius: 2px;
    }
    
    .update-notification .update-btn:hover,
    .update-notification .dismiss-btn:hover {
        background: rgba(255,255,255,0.3);
    }
    
    /* Toast styles */
    .toast {
        font-size: 14px;
        font-weight: 500;
        border-radius: 6px;
    }
    
    /* High DPI optimizations */
    .high-dpi .btn,
    .high-dpi .card {
        border-width: 0.5px;
    }
    
    /* Landscape orientation adjustments */
    .landscape .install-button {
        bottom: 10px;
        right: 10px;
    }
    
    .landscape #offline-indicator {
        padding: 8px;
        font-size: 13px;
    }
    
    /* Reduced motion for accessibility */
    @media (prefers-reduced-motion: reduce) {
        .touch-device .btn:active,
        .touch-device .card:active {
            transform: none;
        }
        
        .install-button,
        .update-notification,
        .toast {
            animation: none !important;
        }
    }
`;

// Add animations to document
document.addEventListener('DOMContentLoaded', () => {
    const style = document.createElement('style');
    style.textContent = mobileAnimations;
    document.head.appendChild(style);
});

// Enhanced Mobile Features Class
class EnhancedMobileFeatures {
    constructor(mobileManager) {
        this.mobileManager = mobileManager;
        this.batterySaver = false;
        this.dataSaver = false;
        
        this.init();
    }
    
    init() {
        this.setupBatteryMonitoring();
        this.setupNetworkMonitoring();
        this.setupPerformanceOptimizations();
        this.setupAccessibilityFeatures();
    }
    
    async setupBatteryMonitoring() {
        if ('getBattery' in navigator) {
            try {
                const battery = await navigator.getBattery();
                
                this.updateBatteryStatus(battery);
                
                battery.addEventListener('levelchange', () => this.updateBatteryStatus(battery));
                battery.addEventListener('chargingchange', () => this.updateBatteryStatus(battery));
            } catch (error) {
                console.log('Battery API not available:', error);
            }
        }
    }
    
    updateBatteryStatus(battery) {
        const level = Math.floor(battery.level * 100);
        const charging = battery.charging;
        
        // Enable battery saver mode when battery is low
        if (level < 20 && !charging) {
            this.enableBatterySaver();
        } else if (level > 30 || charging) {
            this.disableBatterySaver();
        }
        
        this.mobileManager.trackEvent('battery_status', {
            level,
            charging,
            batterySaver: this.batterySaver
        });
    }
    
    enableBatterySaver() {
        if (this.batterySaver) return;
        
        this.batterySaver = true;
        document.body.classList.add('battery-saver');
        
        // Reduce animation frequency
        document.documentElement.style.setProperty('--animation-duration', '0.1s');
        
        // Reduce update frequency
        this.reduceUpdateFrequency();
        
        this.mobileManager.showToast('Battery saver enabled', 'info');
    }
    
    disableBatterySaver() {
        if (!this.batterySaver) return;
        
        this.batterySaver = false;
        document.body.classList.remove('battery-saver');
        
        // Restore normal animation duration
        document.documentElement.style.setProperty('--animation-duration', '0.3s');
        
        // Restore normal update frequency
        this.restoreUpdateFrequency();
    }
    
    setupNetworkMonitoring() {
        if ('connection' in navigator) {
            const connection = navigator.connection;
            
            this.updateNetworkStatus(connection);
            
            connection.addEventListener('change', () => {
                this.updateNetworkStatus(connection);
            });
        }
    }
    
    updateNetworkStatus(connection) {
        const effectiveType = connection.effectiveType;
        const downlink = connection.downlink;
        
        // Enable data saver on slow connections
        if (effectiveType === 'slow-2g' || effectiveType === '2g' || downlink < 1) {
            this.enableDataSaver();
        } else {
            this.disableDataSaver();
        }
        
        this.mobileManager.trackEvent('network_change', {
            effectiveType,
            downlink,
            dataSaver: this.dataSaver
        });
    }
    
    enableDataSaver() {
        if (this.dataSaver) return;
        
        this.dataSaver = true;
        document.body.classList.add('data-saver');
        
        // Reduce image quality
        this.optimizeImages();
        
        // Reduce polling frequency
        this.reducePollingFrequency();
        
        this.mobileManager.showToast('Data saver enabled', 'info');
    }
    
    disableDataSaver() {
        if (!this.dataSaver) return;
        
        this.dataSaver = false;
        document.body.classList.remove('data-saver');
        
        // Restore image quality
        this.restoreImages();
        
        // Restore polling frequency
        this.restorePollingFrequency();
    }
    
    optimizeImages() {
        const images = document.querySelectorAll('img');
        images.forEach(img => {
            if (!img.dataset.originalSrc) {
                img.dataset.originalSrc = img.src;
            }
            
            // Use lower quality or smaller images
            if (img.src.includes('high-res')) {
                img.src = img.src.replace('high-res', 'low-res');
            }
        });
    }
    
    restoreImages() {
        const images = document.querySelectorAll('img[data-original-src]');
        images.forEach(img => {
            img.src = img.dataset.originalSrc;
        });
    }
    
    setupPerformanceOptimizations() {
        // Lazy loading for heavy content
        this.setupLazyLoading();
        
        // Intersection observer for animations
        this.setupAnimationOptimizations();
        
        // Memory management
        this.setupMemoryManagement();
    }
    
    setupLazyLoading() {
        if ('IntersectionObserver' in window) {
            const lazyObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const target = entry.target;
                        
                        if (target.dataset.src) {
                            target.src = target.dataset.src;
                            target.removeAttribute('data-src');
                        }
                        
                        if (target.dataset.load) {
                            this.loadHeavyContent(target);
                        }
                        
                        lazyObserver.unobserve(target);
                    }
                });
            }, { threshold: 0.1 });
            
            // Observe all lazy-loadable elements
            document.querySelectorAll('[data-src], [data-load]').forEach(el => {
                lazyObserver.observe(el);
            });
        }
    }
    
    setupAnimationOptimizations() {
        if ('IntersectionObserver' in window) {
            const animationObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    const target = entry.target;
                    
                    if (entry.isIntersecting) {
                        target.classList.add('animate');
                    } else {
                        target.classList.remove('animate');
                    }
                });
            }, { threshold: 0.1 });
            
            // Observe animation elements
            document.querySelectorAll('.animate-on-scroll').forEach(el => {
                animationObserver.observe(el);
            });
        }
    }
    
    setupMemoryManagement() {
        // Clean up event listeners on page unload
        window.addEventListener('beforeunload', () => {
            this.cleanup();
        });
        
        // Periodic memory cleanup
        setInterval(() => {
            this.performMemoryCleanup();
        }, 60000); // Every minute
    }
    
    performMemoryCleanup() {
        // Remove unused cached data
        if (this.mobileManager.cache) {
            this.mobileManager.cache.cleanup();
        }
        
        // Clear old offline data
        const offlineData = this.mobileManager.getOfflineData();
        const oneWeekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
        
        const filteredData = offlineData.filter(item => item.timestamp > oneWeekAgo);
        if (filteredData.length !== offlineData.length) {
            localStorage.setItem('offline_data', JSON.stringify(filteredData));
        }
    }
    
    setupAccessibilityFeatures() {
        // Respect user's motion preferences
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            document.body.classList.add('reduced-motion');
        }
        
        // Respect user's color scheme preference
        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.body.classList.add('dark-mode');
        }
        
        // High contrast support
        if (window.matchMedia('(prefers-contrast: high)').matches) {
            document.body.classList.add('high-contrast');
        }
        
        // Listen for preference changes
        window.matchMedia('(prefers-reduced-motion: reduce)').addListener((e) => {
            document.body.classList.toggle('reduced-motion', e.matches);
        });
        
        window.matchMedia('(prefers-color-scheme: dark)').addListener((e) => {
            document.body.classList.toggle('dark-mode', e.matches);
        });
    }
    
    cleanup() {
        // Remove event listeners and clean up resources
        console.log('Cleaning up mobile features');
    }
    
    reduceUpdateFrequency() {
        // Reduce the frequency of real-time updates
        if (window.updateInterval) {
            clearInterval(window.updateInterval);
            window.updateInterval = setInterval(window.updateFunction || (() => {}), 5000); // 5 seconds instead of 1
        }
    }
    
    restoreUpdateFrequency() {
        // Restore normal update frequency
        if (window.updateInterval) {
            clearInterval(window.updateInterval);
            window.updateInterval = setInterval(window.updateFunction || (() => {}), 1000); // Back to 1 second
        }
    }
    
    reducePollingFrequency() {
        // Reduce API polling frequency
        document.dispatchEvent(new CustomEvent('dataReducePolling', { detail: { interval: 10000 } }));
    }
    
    restorePollingFrequency() {
        // Restore normal API polling frequency
        document.dispatchEvent(new CustomEvent('dataRestorePolling', { detail: { interval: 3000 } }));
    }
    
    loadHeavyContent(element) {
        // Load heavy content like charts, 3D visualizations, etc.
        const contentType = element.dataset.load;
        
        switch (contentType) {
            case 'chart':
                this.loadChart(element);
                break;
            case '3d-visualization':
                this.load3DVisualization(element);
                break;
            case 'experiment-data':
                this.loadExperimentData(element);
                break;
            default:
                console.log('Unknown content type:', contentType);
        }
    }
    
    loadChart(element) {
        // Placeholder for chart loading
        element.innerHTML = '<div class="chart-placeholder">📊 Chart loading...</div>';
        
        // Simulate chart loading
        setTimeout(() => {
            element.innerHTML = '<div class="chart-loaded">Chart loaded successfully!</div>';
            element.classList.add('content-loaded');
        }, 1000);
    }
    
    load3DVisualization(element) {
        // Placeholder for 3D visualization loading
        element.innerHTML = '<div class="viz-placeholder">🌌 3D Visualization loading...</div>';
        
        // Check if device can handle 3D
        if (this.mobileManager.deviceInfo.isMobile && !this.mobileManager.deviceInfo.isTablet) {
            element.innerHTML = '<div class="viz-fallback">📱 3D view not available on mobile</div>';
            return;
        }
        
        // Simulate 3D loading
        setTimeout(() => {
            element.innerHTML = '<div class="viz-loaded">3D Visualization loaded!</div>';
            element.classList.add('content-loaded');
        }, 2000);
    }
    
    loadExperimentData(element) {
        // Load experiment data dynamically
        const experimentId = element.dataset.experimentId;
        
        if (!experimentId) {
            element.innerHTML = '<div class="error">No experiment ID specified</div>';
            return;
        }
        
        element.innerHTML = '<div class="loading">⚡ Loading experiment data...</div>';
        
        // Fetch experiment data
        fetch(`/api/experiments/${experimentId}/summary`)
            .then(response => response.json())
            .then(data => {
                element.innerHTML = `
                    <div class="experiment-summary">
                        <h4>${data.name || 'Experiment'}</h4>
                        <p>Generation: ${data.current_generation || 0}/${data.total_generations || 0}</p>
                        <p>Best Fitness: ${data.best_fitness || 'N/A'}</p>
                    </div>
                `;
                element.classList.add('content-loaded');
            })
            .catch(error => {
                element.innerHTML = '<div class="error">Failed to load experiment data</div>';
                console.error('Failed to load experiment data:', error);
            });
    }
}

// Initialize enhanced features when mobile manager is ready
document.addEventListener('DOMContentLoaded', () => {
    if (window.mobileManager) {
        window.enhancedMobileFeatures = new EnhancedMobileFeatures(window.mobileManager);
    }
});
