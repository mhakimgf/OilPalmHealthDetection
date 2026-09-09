// Interactive Web GIS Application Logic
document.addEventListener('DOMContentLoaded', () => {
    // 1. Inisialisasi Map Leaflet di Rambah Hilir, Riau
    const defaultCenter = [0.9895, 100.3995];
    const map = L.map('map', {
        center: defaultCenter,
        zoom: 15,
        zoomControl: false
    });

    // Zoom control di pojok kanan atas
    L.control.zoom({ position: 'topright' }).addTo(map);

    // CARTO API Key dari pengguna
    const CARTO_API_KEY = 'cb1_322a_1_ade8349609fe9c1ecd181774';

    // 2. Basemaps (Esri, Google, CARTO Dark, CARTO Voyager, OSM)
    const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Esri World Imagery',
        maxZoom: 19
    }).addTo(map);

    const googleSatellite = L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
        attribution: 'Google Satellite',
        maxZoom: 20
    });

    const cartoDark = L.tileLayer(`https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?api_key=${CARTO_API_KEY}`, {
        attribution: '&copy; CARTO &copy; OpenStreetMap',
        subdomains: 'abcd',
        maxZoom: 20
    });

    const cartoVoyager = L.tileLayer(`https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png?api_key=${CARTO_API_KEY}`, {
        attribution: '&copy; CARTO &copy; OpenStreetMap',
        subdomains: 'abcd',
        maxZoom: 20
    });

    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap'
    });

    // Layer switcher di pojok kanan atas
    L.control.layers({
        '🛰️ Citra Satelit Esri': esriSatellite,
        '🛰️ Citra Satelit Google': googleSatellite,
        '🌙 CARTO Dark Matter': cartoDark,
        '🗺️ CARTO Voyager': cartoVoyager,
        '🗺️ OpenStreetMap': osm
    }, null, { position: 'topright' }).addTo(map);

    // Pastikan ukuran peta terhitung sempurna setelah render
    setTimeout(() => {
        map.invalidateSize();
    }, 200);

    // 3. Geoman Drawing Controls (Polygon)
    map.pm.addControls({
        position: 'topleft',
        drawMarker: false,
        drawCircleMarker: false,
        drawPolyline: false,
        drawRectangle: true,
        drawPolygon: true,
        drawCircle: false,
        drawText: false,
        cutPolygon: false,
        removalMode: true
    });

    // Layer Groups
    let drawnPolygonLayer = null;
    let currentPolygonCoords = null;
    const pointsLayerGroup = L.layerGroup().addTo(map);
    let heatmapLayer = null;
    let lastResultData = null;

    // UI Elements
    const btnDraw = document.getElementById('btnDraw');
    const btnPreset = document.getElementById('btnPreset');
    const btnClear = document.getElementById('btnClear');
    const btnRun = document.getElementById('btnRun');
    const selectYear = document.getElementById('selectYear');
    const polygonStatus = document.getElementById('polygonStatus');
    const loadingBox = document.getElementById('loadingBox');
    const loadingStep = document.getElementById('loadingStep');
    const resultsCard = document.getElementById('resultsCard');
    const btnExportCSV = document.getElementById('btnExportCSV');
    const toggleHeatmap = document.getElementById('toggleHeatmap');

    // Handle Geoman Draw Event
    map.on('pm:create', (e) => {
        if (drawnPolygonLayer) {
            map.removeLayer(drawnPolygonLayer);
        }
        drawnPolygonLayer = e.layer;
        extractCoordsFromLayer(drawnPolygonLayer);
    });

    function extractCoordsFromLayer(layer) {
        const geojson = layer.toGeoJSON();
        currentPolygonCoords = geojson.geometry.coordinates;

        polygonStatus.className = 'polygon-status ready';
        polygonStatus.innerHTML = `<i class="fa-solid fa-circle-check"></i> Poligon Siap (${currentPolygonCoords[0].length} titik sudut)`;
        btnRun.disabled = false;
    }

    // Button: Gambar Poligon
    btnDraw.addEventListener('click', () => {
        map.pm.enableDraw('Polygon', {
            snappable: true,
            snapDistance: 20
        });
    });

    // Button: Contoh Poligon Kebun (Preset Rambah Hilir)
    btnPreset.addEventListener('click', () => {
        if (drawnPolygonLayer) {
            map.removeLayer(drawnPolygonLayer);
        }

        // Koordinat poligon contoh kebun sawit dewasa Rambah Hilir
        const presetCoords = [
            [100.39848, 0.98849],
            [100.40052, 0.98849],
            [100.40052, 0.99068],
            [100.39848, 0.99068],
            [100.39848, 0.98849]
        ];

        // Format GeoJSON coordinates: [[[lon, lat], ...]]
        currentPolygonCoords = [presetCoords];

        // Convert to Leaflet latlngs: [[lat, lon], ...]
        const latlngs = presetCoords.map(c => [c[1], c[0]]);

        drawnPolygonLayer = L.polygon(latlngs, {
            color: '#3b82f6',
            weight: 3,
            fillColor: '#3b82f6',
            fillOpacity: 0.2
        }).addTo(map);

        map.fitBounds(drawnPolygonLayer.getBounds(), { padding: [40, 40] });

        polygonStatus.className = 'polygon-status ready';
        polygonStatus.innerHTML = `<i class="fa-solid fa-circle-check"></i> Contoh Kebun Sawit Terpilih (200m x 240m)`;
        btnRun.disabled = false;
    });

    // Button: Hapus Poligon
    btnClear.addEventListener('click', () => {
        if (drawnPolygonLayer) {
            map.removeLayer(drawnPolygonLayer);
            drawnPolygonLayer = null;
        }
        currentPolygonCoords = null;
        pointsLayerGroup.clearLayers();
        if (heatmapLayer) {
            map.removeLayer(heatmapLayer);
            heatmapLayer = null;
        }
        polygonStatus.className = 'polygon-status empty';
        polygonStatus.innerHTML = `<i class="fa-solid fa-circle-info"></i> Belum ada poligon yang dipilih.`;
        btnRun.disabled = true;
        resultsCard.style.display = 'none';
        lastResultData = null;
    });

    // Button: Jalankan Analisis
    btnRun.addEventListener('click', async () => {
        if (!currentPolygonCoords) return;

        // UI Loading State
        btnRun.disabled = true;
        loadingBox.style.display = 'flex';
        resultsCard.style.display = 'none';
        pointsLayerGroup.clearLayers();
        if (heatmapLayer) {
            map.removeLayer(heatmapLayer);
            heatmapLayer = null;
        }

        const year = selectYear.value;

        // Step simulation text
        loadingStep.innerText = "1/3 Mengakses server Google Earth Engine...";
        setTimeout(() => {
            loadingStep.innerText = "2/3 Memfilter citra Sentinel-2 bebas awan & menghitung Red Edge...";
        }, 1500);
        setTimeout(() => {
            loadingStep.innerText = "3/3 Menjalankan Pipeline 2-Layer AI & Health Scoring...";
        }, 3000);

        try {
            const response = await fetch('/api/predict_polygon', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    coordinates: currentPolygonCoords,
                    year: year
                })
            });

            const res = await response.json();
            loadingBox.style.display = 'none';
            btnRun.disabled = false;

            if (res.success && res.data) {
                lastResultData = res.data;
                renderResultsOnMap(res.data);
                renderResultsInSidebar(res.data);
            } else {
                alert('Gagal memproses poligon: ' + (res.error || 'Terjadi kesalahan'));
            }
        } catch (err) {
            loadingBox.style.display = 'none';
            btnRun.disabled = false;
            alert('Koneksi ke backend gagal: ' + err.message);
        }
    });

    // Render Point Markers & Heatmap
    function renderResultsOnMap(data) {
        pointsLayerGroup.clearLayers();
        const sickPoints = [];

        data.points.forEach(pt => {
            const isSick = pt.status.includes('SAWIT SAKIT');
            const isFiltered = pt.status.includes('Bukan') || pt.status.includes('Non-Sawit');

            const popupHtml = `
                <div style="font-size: 12px; line-height: 1.5; min-width: 170px;">
                    <b style="color: ${pt.color}; font-size: 13px;">${pt.status}</b><br>
                    <hr style="margin: 4px 0; border: 0; border-top: 1px solid #ccc;">
                    ${pt.vigor !== null ? `<b>Vigor Score:</b> ${pt.vigor}<br>` : ''}
                    <b>NDRE (Klorofil):</b> ${pt.ndre}<br>
                    <b>NDVI (Biomassa):</b> ${pt.ndvi}<br>
                    <b>NDBI (Bangunan/Jalan):</b> ${pt.ndbi || 'N/A'}<br>
                    <b>MSI (Stres Air):</b> ${pt.msi}<br>
                    <b>GPS:</b> ${pt.lat.toFixed(5)}, ${pt.lon.toFixed(5)}
                </div>
            `;

            const marker = L.circleMarker([pt.lat, pt.lon], {
                radius: isSick ? 6.5 : (isFiltered ? 2.5 : 4),
                color: pt.color,
                fillColor: pt.color,
                fillOpacity: isSick ? 0.95 : (isFiltered ? 0.35 : 0.8),
                weight: isSick ? 2.5 : (isFiltered ? 0.5 : 1)
            }).bindPopup(popupHtml);

            pointsLayerGroup.addLayer(marker);

            // HANYA titik SAWIT SAKIT sejati yang masuk ke dalam heatmap!
            if (isSick) {
                sickPoints.push([pt.lat, pt.lon, 1.0]);
            }
        });

        // HeatMap Kluster Penyakit
        if (sickPoints.length > 0) {
            heatmapLayer = L.heatLayer(sickPoints, {
                radius: 22,
                blur: 15,
                maxZoom: 17,
                gradient: { 0.2: '#3b82f6', 0.5: '#facc15', 0.8: '#f97316', 1.0: '#ef4444' }
            });

            if (toggleHeatmap.checked) {
                heatmapLayer.addTo(map);
            }
        }
    }

    // Toggle Heatmap Layer
    toggleHeatmap.addEventListener('change', (e) => {
        if (!heatmapLayer) return;
        if (e.target.checked) {
            heatmapLayer.addTo(map);
        } else {
            map.removeLayer(heatmapLayer);
        }
    });

    // Render Sidebar KPI Dashboard
    function renderResultsInSidebar(data) {
        const s = data.summary;

        document.getElementById('valArea').innerText = `${s.area_ha} Ha`;
        document.getElementById('valPoints').innerText = `${s.palm_points} Pkt`;

        document.getElementById('valHealthyPct').innerText = `${s.healthy_pct}% (${s.healthy_count})`;
        document.getElementById('valStressedPct').innerText = `${s.stressed_pct}% (${s.stressed_count})`;
        document.getElementById('valSickPct').innerText = `${s.sick_pct}% (${s.sick_count})`;
        
        const filteredElem = document.getElementById('valFilteredPoints');
        if (filteredElem) {
            filteredElem.innerText = `${s.filtered_points || 0} Pkt`;
        }

        document.getElementById('barHealthy').style.width = `${s.healthy_pct}%`;
        document.getElementById('barStressed').style.width = `${s.stressed_pct}%`;
        document.getElementById('barSick').style.width = `${s.sick_pct}%`;

        const badge = document.getElementById('badgeAlert');
        if (badge) {
            badge.className = `alert-badge ${s.alert_class}`;
            badge.innerText = s.alert;
        }

        // Tampilkan metrik waktu inferensi Machine Learning
        if (s.ml_timing) {
            const mlElem = document.getElementById('valMlTime');
            const geeElem = document.getElementById('valGeeTime');
            const msPointElem = document.getElementById('valMsPerPoint');
            
            if (mlElem) mlElem.innerText = `${s.ml_timing.elapsed_sec}s (${s.ml_timing.elapsed_ms} ms)`;
            if (geeElem) geeElem.innerText = `${s.gee_time_sec || 0}s`;
            if (msPointElem) msPointElem.innerText = `${s.ml_timing.ms_per_sample} ms/pohon`;
            console.log(`>> [ML Benchmark] ${s.model_engine}: ${s.ml_timing.total_samples} points in ${s.ml_timing.elapsed_ms} ms (${s.ml_timing.ms_per_sample} ms/point)`);
        }

        resultsCard.style.display = 'block';
    }

    // Export CSV Download
    btnExportCSV.addEventListener('click', () => {
        if (!lastResultData) return;

        const sickPoints = lastResultData.points.filter(p => p.status.includes('SAWIT SAKIT'));
        if (sickPoints.length === 0) {
            alert('Tidak ada pohon sakit yang terdeteksi di blok ini.');
            return;
        }

        let csvContent = "data:text/csv;charset=utf-8,Latitude,Longitude,Vigor_Score,NDRE,NDVI,MSI,Status\n";
        sickPoints.forEach(p => {
            csvContent += `${p.lat},${p.lon},${p.vigor},${p.ndre},${p.ndvi},${p.msi},${p.status}\n`;
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `target_gps_sawit_sakit_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
