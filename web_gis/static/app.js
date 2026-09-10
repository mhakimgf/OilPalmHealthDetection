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
        // Style permanen untuk poligon teranalisis (biru laut dengan kursor interaktif)
        polyLayer.setStyle({
            color: '#38bdf8',
            weight: 3.5,
            dashArray: null,
            fillColor: '#38bdf8',
            fillOpacity: 0.1
        });

        // Layer group tersendiri untuk titik-titik di blok ini
        const blockPointsGroup = L.layerGroup().addTo(map);
        const blockSickPoints = [];

        data.points.forEach(pt => {
            const isSick = pt.status.includes('SAWIT SAKIT');
            const isFiltered = pt.status.includes('Bukan') || pt.status.includes('Non-Sawit');

            const popupHtml = `
                <div style="font-size: 12px; line-height: 1.5; min-width: 175px;">
                    <div style="font-size: 11px; color: #64748b; font-weight: 600; margin-bottom: 2px;">
                        <i class="fa-solid fa-draw-polygon"></i> Blok #${blockId}
                    </div>
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

            // Hentikan bubbling klik agar tidak memicu penghapusan poligon
            marker.on('click', (e) => {
                L.DomEvent.stopPropagation(e);
            });

            blockPointsGroup.addLayer(marker);

            if (isSick) {
                blockSickPoints.push([pt.lat, pt.lon, 1.0]);
            }
        });

        // Tooltip pada garis poligon
        polyLayer.bindTooltip(`
            <div style="font-size: 12px; line-height: 1.4;">
                <b style="color: #38bdf8;">Blok #${blockId}</b> (${data.summary.area_ha} Ha &bull; ${data.summary.palm_points} Pohon)<br>
                <span style="color: #ef4444; font-weight: 600;">
                    <i class="fa-solid fa-trash"></i> Klik garis poligon untuk hapus blok ini
                </span>
            </div>
        `, { sticky: true });

        // Efek Hover Garis Poligon: Berubah merah dan menebal agar pengguna tahu bisa diklik untuk dihapus
        polyLayer.on('mouseover', () => {
            polyLayer.setStyle({
                color: '#ef4444',
                weight: 5,
                fillOpacity: 0.2
            });
        });

        polyLayer.on('mouseout', () => {
            polyLayer.setStyle({
                color: '#38bdf8',
                weight: 3.5,
                fillOpacity: 0.1
            });
        });

        // Event Klik Garis Poligon: Hapus poligon tersebut beserta titik prediksinya
        polyLayer.on('click', (e) => {
            L.DomEvent.stopPropagation(e);
            const msg = `Hapus Poligon Blok #${blockId} (${data.summary.area_ha} Ha) beserta ${data.summary.palm_points} titik prediksinya?`;
            if (confirm(msg)) {
                deleteAnalyzedBlock(blockId, true);
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
        polygonStatus.innerHTML = `<i class="fa-solid fa-circle-check text-success"></i> <b>Blok #${blockId} Selesai.</b> Total <b>${analyzedBlocks.length} blok aktif</b> di peta.<br><small style="color:var(--text-muted);">Gambar poligon lain untuk menambah blok tanpa menghapus hasil ini.</small>`;
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
        });

        const healthyPct = totalPoints > 0 ? ((totalHealthy / totalPoints) * 100).toFixed(1) : 0;
        const stressedPct = totalPoints > 0 ? ((totalStressed / totalPoints) * 100).toFixed(1) : 0;
        const sickPct = totalPoints > 0 ? ((totalSick / totalPoints) * 100).toFixed(1) : 0;

        if (valBlockCount) {
            valBlockCount.innerText = `${analyzedBlocks.length} Blok Aktif`;
        }

        document.getElementById('valArea').innerText = `${totalArea.toFixed(2)} Ha`;
        document.getElementById('valPoints').innerText = `${totalPoints} Pkt`;

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

        resultsCard.style.display = 'block';
    }

    // Export CSV Download untuk SEMUA Blok Aktif
    btnExportCSV.addEventListener('click', () => {
        if (analyzedBlocks.length === 0) {
            alert('Tidak ada blok kebun aktif.');
            return;
        }

        const allSickPoints = [];
        analyzedBlocks.forEach(b => {
            const sicks = b.data.points.filter(p => p.status.includes('SAWIT SAKIT'));
            sicks.forEach(p => {
                allSickPoints.push({
                    blockId: b.id,
                    ...p
                });
            });
        });

        if (allSickPoints.length === 0) {
            alert('Tidak ada pohon sakit yang terdeteksi di seluruh blok aktif.');
            return;
        }

        let csvContent = "data:text/csv;charset=utf-8,Blok_ID,Latitude,Longitude,Vigor_Score,NDRE,NDVI,MSI,Status\n";
        allSickPoints.forEach(p => {
            csvContent += `Blok #${p.blockId},${p.lat},${p.lon},${p.vigor},${p.ndre},${p.ndvi},${p.msi},"${p.status}"\n`;
        });

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `target_gps_sawit_sakit_${analyzedBlocks.length}blok_${Date.now()}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
