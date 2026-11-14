/* ThinkStep service worker.
   Its only job: let ThinkStep be installed like an app, and show a friendly
   "you're offline" page instead of the browser's error when there's no
   internet. It never saves chats, answers or anything from /api/ — those
   always go straight to the server, so kids always get fresh, live data. */
const CACHE = 'thinkstep-shell-v1';
const OFFLINE_URL = '/static/offline.html';
const PRECACHE = [OFFLINE_URL, '/static/icon-192.png', '/static/favicon.svg'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  // Only page loads get the offline fallback. Everything else (API calls,
  // pictures, fonts) is left to the browser exactly as before.
  if (req.method !== 'GET' || req.mode !== 'navigate') return;
  event.respondWith(
    fetch(req).catch(() => caches.match(OFFLINE_URL).then((r) => r || Response.error()))
  );
});
