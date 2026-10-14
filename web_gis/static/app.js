// Interactive Web GIS Application Logic (Multi-Polygon & Layer Management)
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

    // Custom Pane khusus Titik Prediksi Sawit (Z-Index 620: di atas poligon 400, di bawah popup 700)
    map.createPane('palmPointsPane');
    map.getPane('palmPointsPane').style.zIndex = '620';

    // 3. Geoman Drawing Controls (Polygon & Rectangle)
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

    // ==========================================
    // Multi-Polygon State Management
    // ==========================================
    let pendingPolygonLayer = null;       // Poligon yang baru digambar/dipilih (menunggu 'btnRun')
    let currentPolygonCoords = null;      // Koordinat GeoJSON poligon pending
    let blockIdCounter = 0;              // Counter penomoran blok (Blok #1, Blok #2, ...)
    let analyzedBlocks = [];             // Array of { id, polygonLayer, pointsGroup, data, sickPoints }
    let heatmapLayer = null;             // Layer Heatmap gabungan semua blok aktif

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
    const valBlockCount = document.getElementById('valBlockCount');

    // Handle Geoman Draw Event: Membuat poligon pending baru
    map.on('pm:create', (e) => {
        // Jika ada poligon pending sebelumnya yang belum dianalisis, hapus yang pending saja
        if (pendingPolygonLayer) {
            map.removeLayer(pendingPolygonLayer);
        }
        pendingPolygonLayer = e.layer;
        
        // Beri style garis putus-putus oranye untuk menandakan poligon sedang pending
        pendingPolygonLayer.setStyle({
            color: '#f59e0b',
            weight: 3,
            dashArray: '6, 6',
            fillColor: '#f59e0b',
            fillOpacity: 0.15
        });

        extractCoordsFromLayer(pendingPolygonLayer);
    });

    // Handle Geoman Removal Event: Sinkronisasi jika dihapus via tool Geoman
    map.on('pm:remove', (e) => {
        const removedLayer = e.layer;
        if (removedLayer === pendingPolygonLayer) {
            pendingPolygonLayer = null;
            currentPolygonCoords = null;
            btnRun.disabled = true;
            polygonStatus.className = 'polygon-status empty';
            polygonStatus.innerHTML = '<i class="fa-solid fa-circle-info"></i> Poligon pending dibatalkan.';
            return;
        }

        const match = analyzedBlocks.find(b => b.polygonLayer === removedLayer);
        if (match) {
            deleteAnalyzedBlock(match.id, false); // false = jangan panggil map.removeLayer lagi karena Geoman sudah menghapusnya
        }
    });

    function extractCoordsFromLayer(layer) {
        const geojson = layer.toGeoJSON();
        currentPolygonCoords = geojson.geometry.coordinates;

        const activeCount = analyzedBlocks.length;
        polygonStatus.className = 'polygon-status ready';
        polygonStatus.innerHTML = `<i class="fa-solid fa-circle-check text-warning"></i> <b>Poligon Baru Siap</b> (${currentPolygonCoords[0].length} sudut).<br><small style="color:var(--text-muted);">${activeCount > 0 ? activeCount + ' blok lain tetap aktif. ' : ''}Klik <b>Jalankan Analisis</b> untuk memproses blok ini.</small>`;
        btnRun.disabled = false;
    }

    // Button: Aktifkan Mode Gambar Poligon
    btnDraw.addEventListener('click', () => {
        map.pm.enableDraw('Polygon', {
            snappable: true,
            snapDistance: 20
        });
    });

    // Button: Contoh Poligon Kebun (Preset Rambah Hilir)
    btnPreset.addEventListener('click', () => {
        // Hapus pending sebelumnya jika ada
        if (pendingPolygonLayer) {
            map.removeLayer(pendingPolygonLayer);
        }

        // Koordinat poligon contoh kebun sawit dewasa Rambah Hilir
        const presetCoords = [
            [100.39848, 0.98849],
            [100.40052, 0.98849],
            [100.40052, 0.99068],
            [100.39848, 0.99068],
            [100.39848, 0.98849]
        ];

        currentPolygonCoords = [presetCoords];
        const latlngs = presetCoords.map(c => [c[1], c[0]]);

        pendingPolygonLayer = L.polygon(latlngs, {
            color: '#f59e0b',
            weight: 3,
            dashArray: '6, 6',
            fillColor: '#f59e0b',
            fillOpacity: 0.15
        }).addTo(map);

        map.fitBounds(pendingPolygonLayer.getBounds(), { padding: [40, 40] });

        const activeCount = analyzedBlocks.length;
        polygonStatus.className = 'polygon-status ready';
        polygonStatus.innerHTML = `<i class="fa-solid fa-circle-check text-warning"></i> <b>Contoh Kebun Terpilih</b> (200m x 240m).<br><small style="color:var(--text-muted);">${activeCount > 0 ? activeCount + ' blok lain tetap aktif. ' : ''}Klik <b>Jalankan Analisis</b> untuk memproses blok ini.</small>`;
        btnRun.disabled = false;
    });

    // Button: Hapus SEMUA Poligon dan Hasil Prediksi (Tombol Tempat Sampah)
    btnClear.addEventListener('click', () => {
        // 1. Hapus pending polygon jika ada
        if (pendingPolygonLayer) {
            map.removeLayer(pendingPolygonLayer);
            pendingPolygonLayer = null;
        }
        currentPolygonCoords = null;

        // 2. Hapus semua analyzed blocks dari peta
        analyzedBlocks.forEach(block => {
            if (block.polygonLayer) map.removeLayer(block.polygonLayer);
            if (block.pointsGroup) map.removeLayer(block.pointsGroup);
        });
        analyzedBlocks = [];

        // 3. Hapus Heatmap
        if (heatmapLayer) {
            map.removeLayer(heatmapLayer);
            heatmapLayer = null;
        }

        // 4. Reset UI Sidebar
        polygonStatus.className = 'polygon-status empty';
        polygonStatus.innerHTML = `<i class="fa-solid fa-circle-info"></i> Seluruh poligon dan hasil prediksi telah dihapus.`;
        btnRun.disabled = true;
        resultsCard.style.display = 'none';
        if (valBlockCount) valBlockCount.innerText = '0 Blok Aktif';
    });

    // Button: Jalankan Analisis untuk Poligon Pending
    btnRun.addEventListener('click', async () => {
        if (!currentPolygonCoords || !pendingPolygonLayer) return;

        // UI Loading State
        btnRun.disabled = true;
        loadingBox.style.display = 'flex';

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

            if (res.success && res.data) {
                // Beri ID unik untuk blok baru
                blockIdCounter++;
                const newBlockId = blockIdCounter;
                
                // Konversi poligon pending menjadi poligon teranalisis permanen
                const analyzedPoly = pendingPolygonLayer;
                pendingPolygonLayer = null;
                currentPolygonCoords = null;

                // Daftarkan blok dan render titik-titiknya
                registerAnalyzedBlock(newBlockId, analyzedPoly, res.data);
            } else {
                btnRun.disabled = false;
                alert('Gagal memproses poligon: ' + (res.error || 'Terjadi kesalahan'));
            }
        } catch (err) {
            loadingBox.style.display = 'none';
            btnRun.disabled = false;
            alert('Koneksi ke backend gagal: ' + err.message);
        }
    });

    // Pendaftaran Blok Hasil Analisis & Pembuatan Layer Titik
    function registerAnalyzedBlock(blockId, polyLayer, data) {
        // Style permanen untuk poligon teranalisis (garis batas kebun rapi, tanpa membajak klik)
        polyLayer.setStyle({
            color: '#38bdf8',
            weight: 2.5,
            dashArray: null,
            fillColor: '#38bdf8',
            fillOpacity: 0.05
        });

        // Bersihkan listener event yang destruktif agar tidak membajak klik titik
        polyLayer.off('click');
        polyLayer.off('mouseover');
        polyLayer.off('mouseout');
        if (polyLayer.unbindTooltip) polyLayer.unbindTooltip();

        // Popup informatif saat garis poligon diklik (tidak langsung menghapus secara mendadak)
        polyLayer.bindPopup(`
            <div style="font-size: 12px; line-height: 1.5; min-width: 170px;">
                <div style="font-size: 13px; font-weight: 700; color: #38bdf8; margin-bottom: 4px;">
                    <i class="fa-solid fa-draw-polygon"></i> Blok #${blockId}
                </div>
                <div>Luas: <b>${data.summary.area_ha} Ha</b></div>
                <div>Total Titik: <b>${data.summary.palm_points} Pohon</b></div>
                ${data.summary.sick_count > 0 ? `<div style="color: #ef4444; font-weight: 600;">Sawit Sakit: ${data.summary.sick_count} Pohon (${data.summary.sick_pct}%)</div>` : ''}
                <hr style="margin: 6px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.15);">
                <button class="btn btn-sm btn-danger-outline" style="width: 100%; padding: 4px 8px; font-size: 11px; cursor: pointer;" onclick="deleteAnalyzedBlock(${blockId})">
                    <i class="fa-solid fa-trash"></i> Hapus Blok Ini
                </button>
            </div>
        `);

        // Layer group tersendiri untuk titik-titik di blok ini
        const blockPointsGroup = L.layerGroup().addTo(map);
        const blockSickPoints = [];

        data.points.forEach(pt => {
            const isTier1 = (pt.priority_tier && pt.priority_tier.includes('Tier 1')) || pt.status.includes('SAWIT SAKIT');
            const isTier2 = (pt.priority_tier && pt.priority_tier.includes('Tier 2')) || pt.status.includes('Stres');
            const isFiltered = pt.status.includes('Bukan') || pt.status.includes('Non-Sawit') || (pt.priority_tier && pt.priority_tier.includes('Filter'));

            const anomScore = pt.anomaly_score !== undefined ? Number(pt.anomaly_score).toFixed(3) : '-';
            const tierBadge = pt.priority_tier || (isTier1 ? '🔴 Tier 1: Karantina' : (isTier2 ? '🟡 Tier 2: Stres' : '🟢 Tier 3: Sehat'));
            const actionText = pt.action || '-';

            const popupHtml = `
                <div style="font-size: 12px; line-height: 1.5; min-width: 220px; max-width: 280px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                        <span style="font-size: 11px; color: #64748b; font-weight: 700;">
                            <i class="fa-solid fa-draw-polygon"></i> Blok #${blockId}
                        </span>
                        <span style="font-size: 10px; background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 4px;">
                            ${pt.lat.toFixed(5)}, ${pt.lon.toFixed(5)}
                        </span>
                    </div>
                    <div style="padding: 4px 8px; border-radius: 4px; background: ${isTier1 ? 'rgba(239,68,68,0.18)' : (isTier2 ? 'rgba(245,158,11,0.18)' : (isFiltered ? 'rgba(148,163,184,0.12)' : 'rgba(34,197,94,0.15)'))}; margin-bottom: 6px;">
                        <b style="color: ${pt.color}; font-size: 12px;">${pt.status}</b>
                    </div>
                    <div style="margin-bottom: 6px; padding: 4px 6px; background: rgba(0,0,0,0.2); border-radius: 4px; font-size: 11px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94a3b8;">Skor Anomali:</span>
                            <b style="color: ${Number(anomScore) >= 0.65 ? '#ef4444' : (Number(anomScore) >= 0.40 ? '#f59e0b' : '#38bdf8')}; font-size: 12px;">${anomScore}</b>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 2px;">
                            <span style="color: #94a3b8;">Prioritas:</span>
                            <span style="font-weight: 600;">${tierBadge}</span>
                        </div>
                    </div>
                    <table style="width: 100%; font-size: 11px; border-collapse: collapse; margin-bottom: 6px;">
                        <tr><td style="color:#94a3b8; padding: 2px 0;">NDRE (Klorofil):</td><td style="font-weight:700; text-align:right;">${pt.ndre}</td></tr>
                        <tr><td style="color:#94a3b8; padding: 2px 0;">NDVI (Biomassa):</td><td style="font-weight:700; text-align:right;">${pt.ndvi}</td></tr>
                        <tr><td style="color:#94a3b8; padding: 2px 0;">MSI (Stres Air):</td><td style="font-weight:700; text-align:right;">${pt.msi}</td></tr>
                        ${pt.ndbi ? `<tr><td style="color:#94a3b8; padding: 2px 0;">NDBI:</td><td style="font-weight:700; text-align:right;">${pt.ndbi}</td></tr>` : ''}
                    </table>
                    ${actionText !== '-' ? `
                    <div style="font-size: 10px; color: #cbd5e1; background: rgba(255,255,255,0.04); padding: 5px 7px; border-left: 2px solid ${pt.color}; border-radius: 2px; line-height: 1.35;">
                        <i class="fa-solid fa-clipboard-check" style="color: ${pt.color};"></i> <b>Tindakan:</b> ${actionText}
                    </div>` : ''}
                </div>
            `;

            const marker = L.circleMarker([pt.lat, pt.lon], {
                pane: 'palmPointsPane', // Menjamin titik SELALU di atas poligon
                radius: isTier1 ? 7.5 : (isTier2 ? 6 : (isFiltered ? 3 : 5)),
                color: isTier1 ? '#ffffff' : (isTier2 ? '#ffffff' : pt.color),
                fillColor: pt.color,
                fillOpacity: isTier1 ? 1.0 : (isTier2 ? 0.9 : (isFiltered ? 0.35 : 0.85)),
                weight: isTier1 ? 2 : (isTier2 ? 1.5 : (isFiltered ? 0.5 : 1)),
                className: 'palm-point-marker'
            }).bindPopup(popupHtml, { autoPan: true });

            // Efek hover interaktif halus pada titik
            marker.on('mouseover', function () {
                this.setRadius(isTier1 ? 9.5 : (isTier2 ? 7.5 : (isFiltered ? 4.5 : 7)));
                this.bringToFront();
            });

            marker.on('mouseout', function () {
                this.setRadius(isTier1 ? 7.5 : (isTier2 ? 6 : (isFiltered ? 3 : 5)));
            });

            // Pastikan event klik pada titik tidak bocor ke peta/poligon
            marker.on('click', (e) => {
                L.DomEvent.stopPropagation(e);
            });

            blockPointsGroup.addLayer(marker);

            // Anomaly Heatmap: Beri bobot intensitas kontinu sesuai skor anomali iForest
            if (pt.anomaly_score !== undefined && pt.anomaly_score >= 0.40 && !isFiltered) {
                blockSickPoints.push([pt.lat, pt.lon, Number(pt.anomaly_score)]);
            } else if (isTier1) {
                blockSickPoints.push([pt.lat, pt.lon, 1.0]);
            }
        });

        // Simpan ke daftar analyzedBlocks
        analyzedBlocks.push({
            id: blockId,
            polygonLayer: polyLayer,
            pointsGroup: blockPointsGroup,
            data: data,
            sickPoints: blockSickPoints
        });

        // Refresh Heatmap & Dashboard Sidebar
        refreshCombinedHeatmap();
        refreshSidebarDashboard();

        polygonStatus.className = 'polygon-status ready';
        polygonStatus.innerHTML = `<i class="fa-solid fa-circle-check text-success"></i> <b>Blok #${blockId} Selesai.</b> Total <b>${analyzedBlocks.length} blok aktif</b>.<br><small style="color:var(--text-muted);">Klik titik mana pun untuk melihat detail spektral atau kelola via Daftar Blok di sidebar.</small>`;
    }

    // Fungsi Hapus Blok Tunggal (Dipanggil saat garis poligon diklik)
    function deleteAnalyzedBlock(blockId, removePolygonFromMap = true) {
        const idx = analyzedBlocks.findIndex(b => b.id === blockId);
        if (idx === -1) return;

        const block = analyzedBlocks[idx];

        // Hapus elemen peta
        if (removePolygonFromMap && block.polygonLayer) {
            map.removeLayer(block.polygonLayer);
        }
        if (block.pointsGroup) {
            map.removeLayer(block.pointsGroup);
        }

        // Hapus dari state
        analyzedBlocks.splice(idx, 1);

        // Refresh Heatmap gabungan & KPI Sidebar
        refreshCombinedHeatmap();
        refreshSidebarDashboard();

        // Update status teks
        if (analyzedBlocks.length === 0) {
            polygonStatus.className = 'polygon-status empty';
            polygonStatus.innerHTML = `<i class="fa-solid fa-circle-info"></i> Blok #${blockId} telah dihapus. Belum ada poligon aktif.`;
            resultsCard.style.display = 'none';
            if (valBlockCount) valBlockCount.innerText = '0 Blok Aktif';
            btnRun.disabled = true;
        } else {
            polygonStatus.className = 'polygon-status ready';
            polygonStatus.innerHTML = `<i class="fa-solid fa-circle-check text-success"></i> Blok #${blockId} dihapus. Tersisa <b>${analyzedBlocks.length} blok aktif</b> di peta.`;
        }
    }

    // Expose ke window agar bisa dipanggil jika diperlukan
    window.deleteAnalyzedBlock = deleteAnalyzedBlock;

    // Refresh Heatmap Gabungan dari SEMUA Blok Aktif
    function refreshCombinedHeatmap() {
        if (heatmapLayer) {
            map.removeLayer(heatmapLayer);
            heatmapLayer = null;
        }

        const allSickPoints = [];
        analyzedBlocks.forEach(b => {
            b.sickPoints.forEach(pt => allSickPoints.push(pt));
        });

        if (allSickPoints.length > 0) {
            heatmapLayer = L.heatLayer(allSickPoints, {
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

    // Render / Refresh Kumulatif KPI Dashboard di Sidebar
    function refreshSidebarDashboard() {
        if (analyzedBlocks.length === 0) {
            resultsCard.style.display = 'none';
            if (valBlockCount) valBlockCount.innerText = '0 Blok Aktif';
            return;
        }

        let totalArea = 0;
        let totalPoints = 0;
        let totalHealthy = 0;
        let totalStressed = 0;
        let totalSick = 0;
        let totalFiltered = 0;
        let totalMlMs = 0;
        let totalGeeSec = 0;
        let totalAnomalySum = 0;
        let totalAnomalyCount = 0;

        analyzedBlocks.forEach(b => {
            const s = b.data.summary;
            totalArea += (s.area_ha || 0);
            totalPoints += (s.palm_points || 0);
            totalHealthy += (s.healthy_count || 0);
            totalStressed += (s.stressed_count || 0);
            totalSick += (s.sick_count || 0);
            totalFiltered += (s.filtered_points || 0);
            if (s.ml_timing) totalMlMs += s.ml_timing.elapsed_ms;
            if (s.gee_time_sec) totalGeeSec += s.gee_time_sec;

            b.data.points.forEach(pt => {
                if (pt.anomaly_score !== undefined && !pt.status.includes('Bukan') && !pt.status.includes('Non-Sawit')) {
                    totalAnomalySum += Number(pt.anomaly_score);
                    totalAnomalyCount++;
                }
            });
        });

        const healthyPct = totalPoints > 0 ? ((totalHealthy / totalPoints) * 100).toFixed(1) : 0;
        const stressedPct = totalPoints > 0 ? ((totalStressed / totalPoints) * 100).toFixed(1) : 0;
        const sickPct = totalPoints > 0 ? ((totalSick / totalPoints) * 100).toFixed(1) : 0;
        const avgAnomaly = totalAnomalyCount > 0 ? (totalAnomalySum / totalAnomalyCount).toFixed(3) : '0.000';

        if (valBlockCount) {
            valBlockCount.innerText = `${analyzedBlocks.length} Blok Aktif`;
        }

        document.getElementById('valArea').innerText = `${totalArea.toFixed(2)} Ha`;
        document.getElementById('valPoints').innerText = `${totalPoints} Pkt`;
        const valAvgElem = document.getElementById('valAvgScore');
        if (valAvgElem) valAvgElem.innerText = avgAnomaly;

        document.getElementById('valHealthyPct').innerText = `${healthyPct}% (${totalHealthy})`;
        document.getElementById('valStressedPct').innerText = `${stressedPct}% (${totalStressed})`;
        document.getElementById('valSickPct').innerText = `${sickPct}% (${totalSick})`;

        const filteredElem = document.getElementById('valFilteredPoints');
        if (filteredElem) {
            filteredElem.innerText = `${totalFiltered} Pkt`;
        }

        document.getElementById('barHealthy').style.width = `${healthyPct}%`;
        document.getElementById('barStressed').style.width = `${stressedPct}%`;
        document.getElementById('barSick').style.width = `${sickPct}%`;

        // Alert Status Kumulatif
        const badge = document.getElementById('badgeAlert');
        if (badge) {
            if (parseFloat(sickPct) > 15) {
                badge.className = 'alert-badge danger';
                badge.innerText = 'BAHAYA EPIDEMI';
            } else if (parseFloat(sickPct) > 0) {
                badge.className = 'alert-badge warning';
                badge.innerText = 'WASPADA KLUSTER';
            } else if (parseFloat(stressedPct) > 20) {
                badge.className = 'alert-badge warning';
                badge.innerText = 'PERLU PEMUPUKAN';
            } else {
                badge.className = 'alert-badge optimal';
                badge.innerText = 'KEBUN OPTIMAL';
            }
        }

        // Tampilkan metrik kecepatan inferensi ML dari blok terbaru
        const latestBlock = analyzedBlocks[analyzedBlocks.length - 1];
        if (latestBlock && latestBlock.data.summary.ml_timing) {
            const s = latestBlock.data.summary;
            const mlElem = document.getElementById('valMlTime');
            const geeElem = document.getElementById('valGeeTime');
            const msPointElem = document.getElementById('valMsPerPoint');

            if (mlElem) mlElem.innerText = `${s.ml_timing.elapsed_sec}s (${s.ml_timing.elapsed_ms} ms)`;
            if (geeElem) geeElem.innerText = `${s.gee_time_sec || 0}s`;
            if (msPointElem) msPointElem.innerText = `${s.ml_timing.ms_per_sample} ms/pohon`;
        }

        // Render daftar blok aktif di sidebar
        renderActiveBlocksList();

        resultsCard.style.display = 'block';
    }

    // Fungsi Render Daftar Blok Aktif di Sidebar
    function renderActiveBlocksList() {
        const listElem = document.getElementById('activeBlocksList');
        if (!listElem) return;
        listElem.innerHTML = '';

        analyzedBlocks.forEach(b => {
            const s = b.data.summary;
            const item = document.createElement('div');
            item.className = 'active-block-item';
            item.innerHTML = `
                <div class="block-info">
                    <span style="font-size: 12px; font-weight: 700; color: #38bdf8;">
                        <i class="fa-solid fa-draw-polygon"></i> Blok #${b.id}
                    </span>
                    <span style="font-size: 11px; color: var(--text-muted);">
                        ${s.area_ha} Ha &bull; ${s.palm_points} Pohon ${s.sick_count > 0 ? `<b style="color:#ef4444;">(${s.sick_count} Sakit)</b>` : ''}
                    </span>
                </div>
                <div class="block-actions">
                    <button class="btn btn-xs btn-outline" title="Fokus / Zoom ke Blok #${b.id}" onclick="zoomToBlock(${b.id})">
                        <i class="fa-solid fa-crosshairs"></i>
                    </button>
                    <button class="btn btn-xs btn-danger-outline" title="Hapus Blok #${b.id}" onclick="confirmDeleteBlock(${b.id})">
                        <i class="fa-solid fa-trash"></i>
                    </button>
                </div>
            `;
            listElem.appendChild(item);
        });
    }

    // Fungsi Zoom ke Poligon Blok Tertentu
    function zoomToBlock(blockId) {
        const b = analyzedBlocks.find(item => item.id === blockId);
        if (b && b.polygonLayer) {
            map.fitBounds(b.polygonLayer.getBounds(), { padding: [40, 40] });
            b.polygonLayer.openPopup();
        }
    }
    window.zoomToBlock = zoomToBlock;

    // Fungsi Konfirmasi Hapus Blok dari Tombol Sidebar
    function confirmDeleteBlock(blockId) {
        const b = analyzedBlocks.find(item => item.id === blockId);
        const name = b ? `Blok #${blockId} (${b.data.summary.area_ha} Ha)` : `Blok #${blockId}`;
        if (confirm(`Hapus ${name} beserta titik-titik prediksinya?`)) {
            deleteAnalyzedBlock(blockId, true);
        }
    }
    window.confirmDeleteBlock = confirmDeleteBlock;

    // Export CSV Download untuk SEMUA Blok Aktif dengan Urutan Prioritas Triage Anomaly
    btnExportCSV.addEventListener('click', () => {
        if (analyzedBlocks.length === 0) {
            alert('Tidak ada blok kebun aktif.');
            return;
        }

        const allTargetPoints = [];
        analyzedBlocks.forEach(b => {
            // Saring titik-titik yang membutuhkan inspeksi (Tier 1 Karantina dan Tier 2 Stres)
            const targets = b.data.points.filter(p => 
                (p.priority_tier && (p.priority_tier.includes('Tier 1') || p.priority_tier.includes('Tier 2'))) ||
                p.status.includes('SAWIT SAKIT') ||
                p.status.includes('Stres')
            );
            targets.forEach(p => {
                allTargetPoints.push({
                    blockId: b.id,
                    ...p
                });
            });
        });

        if (allTargetPoints.length === 0) {
            alert('Tidak ada pohon sakit atau stres yang terdeteksi di seluruh blok aktif.');
            return;
        }

        // URUTKAN BERDASARKAN SKOR ANOMALI TERTINGGI (URGENT FIRST)
        allTargetPoints.sort((a, b) => (Number(b.anomaly_score) || 0) - (Number(a.anomaly_score) || 0));

        let csvContent = "data:text/csv;charset=utf-8,Ranking_Prioritas,Blok_ID,Latitude,Longitude,Skor_Anomali,Tier_Prioritas,Status_Diagnosa,Rekomendasi_Tindakan,NDRE,NDVI,MSI\n";
        allTargetPoints.forEach((p, idx) => {
            const rank = idx + 1;
            const score = p.anomaly_score !== undefined ? Number(p.anomaly_score).toFixed(4) : '0.0000';
            const tier = p.priority_tier || 'Tier 1';
            const action = (p.action || '-').replace(/"/g, '""');
            csvContent += `${rank},Blok #${p.blockId},${p.lat},${p.lon},${score},"${tier}","${p.status}","${action}",${p.ndre},${p.ndvi},${p.msi}\n`;
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `target_triage_gps_iforest_${analyzedBlocks.length}blok_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
