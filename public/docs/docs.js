(async function init() {
  const preferred = localStorage.getItem('c4e_container') || 'initial';

  async function loadManifest() {
    const res = await fetch('/docs/apis.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Manifest ${res.status}`);
    const data = await res.json();
    if (!Array.isArray(data) || data.length === 0) throw new Error('Manifest leer');
    // Normalisieren: {name, url}
    return data
      .map(e => ({ name: String(e.name || '').trim(), url: String(e.url || '').trim() }))
      .filter(e => e.name && e.url);
  }

  const entries = await loadManifest();
  const services = entries.map(e => e.name);
  const urls = entries;
  const primary = services.includes(preferred) ? preferred : services[0];

  const ui = SwaggerUIBundle({
    urls,
    'urls.primaryName': primary,
    dom_id: '#swagger-ui',
    deepLinking: true,
    docExpansion: 'list',
    defaultModelsExpandDepth: -1,
    defaultModelExpandDepth: 0,
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
    layout: 'StandaloneLayout',
    filter: true,
  });

  // Auswahländerung in der Topbar persistieren
  const hookSelect = () => {
    const sel = document.querySelector('.topbar select');
    if (!sel) { setTimeout(hookSelect, 500); return; }
    sel.addEventListener('change', () => {
      const name = sel.options[sel.selectedIndex]?.text?.trim();
      if (name) localStorage.setItem('c4e_container', name);
    });
  };
  hookSelect();
})().catch(err => {
  // Optional: Einfache Fehlermeldung ausgeben (ohne Fallback-Logik)
  const host = document.getElementById('swagger-ui');
  if (host) host.innerHTML = `<div style="padding:16px;font-family:system-ui">Fehler beim Laden der API-Liste: ${err.message}</div>`;
});
