/* Cross-Origin Isolation Service Worker fallback for static hosts without COOP/COEP headers */
self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') {
    return;
  }

  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.status === 0 || response.type === 'opaque') {
          return response;
        }

        const headers = new Headers(response.headers);
        const url = new URL(request.url);
        const isSameOrigin = url.origin === self.location.origin;

        if (isSameOrigin) {
          headers.set('Cross-Origin-Opener-Policy', 'same-origin');
          headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
          headers.set('Cross-Origin-Resource-Policy', 'same-origin');
        } else if (!headers.has('Cross-Origin-Resource-Policy')) {
          headers.set('Cross-Origin-Resource-Policy', 'cross-origin');
        }

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
      })
      .catch((err) => {
        throw err;
      })
  );
});
