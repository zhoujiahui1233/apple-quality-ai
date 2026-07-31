self.addEventListener("fetch", function(e) {
  e.respondWith(fetch(e.request).catch(function() {
    return new Response("", { status: 503 });
  }));
});
