/* ThinkStep's "Get the app" feature was removed. This file only switches off
   the helper for anyone who had it: it deletes what it saved and removes
   itself, so pages load straight from the server like before. */
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.map((k) => caches.delete(k)));
    await self.registration.unregister();
  })());
});
