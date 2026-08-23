(function () {
  const WORLD_CENTER = [20, 0];
  const WORLD_ZOOM = 2;
  const PLACE_ZOOM = 8;
  const LIGHT_TILES = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";
  const DARK_TILES = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
  const TILE_ATTR =
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>';

  let map;
  let lightLayer;
  let darkLayer;
  let markers = [];

  function escapeHtml(value) {
    return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function currentTheme() {
    if (typeof determineComputedTheme === "function") {
      return determineComputedTheme();
    }
    return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }

  function markerColor(kind) {
    return kind === "trip" ? "#d64545" : "#3b6fd8";
  }

  function markerStyle(kind) {
    return {
      radius: 8,
      color: "#ffffff",
      weight: 2,
      fillColor: markerColor(kind),
      fillOpacity: 0.92,
    };
  }

  function popupHtml(place) {
    const name = escapeHtml(place.name || "");
    const note = place.note ? `<p>${escapeHtml(place.note)}</p>` : "";
    return `<div class="travel-map__popup"><strong>${name}</strong>${note}</div>`;
  }

  function setActivePlace(index) {
    document.querySelectorAll(".travel-map__place").forEach((button) => {
      const isActive = Number(button.dataset.index) === index;
      button.classList.toggle("is-active", isActive);
      button.setAttribute("aria-current", isActive ? "true" : "false");
    });
  }

  function tileLayer(url) {
    return L.tileLayer(url, {
      attribution: TILE_ATTR,
      subdomains: "abcd",
      maxZoom: 19,
      minZoom: 2,
    });
  }

  window.setTravelMapTheme = function (theme) {
    if (!map) {
      return;
    }
    const dark = theme === "dark";
    if (dark) {
      if (map.hasLayer(lightLayer)) {
        map.removeLayer(lightLayer);
      }
      if (!map.hasLayer(darkLayer)) {
        darkLayer.addTo(map);
      }
    } else {
      if (map.hasLayer(darkLayer)) {
        map.removeLayer(darkLayer);
      }
      if (!map.hasLayer(lightLayer)) {
        lightLayer.addTo(map);
      }
    }
    markers.forEach((marker) => marker.setStyle(markerStyle(marker._placeKind)));
  };

  function goToPlace(index, openPopup) {
    const marker = markers[index];
    if (!marker) {
      return;
    }
    const latlng = marker.getLatLng();
    map.flyTo(latlng, Math.max(map.getZoom(), PLACE_ZOOM), { duration: 0.7 });
    if (openPopup) {
      marker.openPopup();
    }
    setActivePlace(index);
  }

  function initTravelMap() {
    const canvas = document.getElementById("travel-map");
    if (!canvas || typeof L === "undefined") {
      return;
    }

    let places = [];
    try {
      places = JSON.parse(canvas.dataset.places || "[]");
    } catch (err) {
      places = [];
    }
    if (!Array.isArray(places)) {
      places = [];
    }

    map = L.map(canvas, {
      center: WORLD_CENTER,
      zoom: WORLD_ZOOM,
      minZoom: 2,
      maxZoom: 19,
      worldCopyJump: true,
      scrollWheelZoom: false,
      zoomControl: true,
    });

    lightLayer = tileLayer(LIGHT_TILES);
    darkLayer = tileLayer(DARK_TILES);
    window.setTravelMapTheme(currentTheme());

    places.forEach((place, index) => {
      const lat = Number(place.lat);
      const lng = Number(place.lng);
      if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
        return;
      }
      const kind = place.kind === "trip" ? "trip" : "work";
      const marker = L.circleMarker([lat, lng], markerStyle(kind)).addTo(map);
      marker._placeKind = kind;
      marker.bindPopup(popupHtml(place), { closeButton: false });
      marker.on("click", function () {
        goToPlace(index, true);
      });
      markers[index] = marker;
    });

    canvas.addEventListener("mouseenter", function () {
      map.scrollWheelZoom.enable();
    });
    canvas.addEventListener("mouseleave", function () {
      map.scrollWheelZoom.disable();
    });

    const reset = document.getElementById("travel-map-reset");
    if (reset) {
      reset.addEventListener("click", function () {
        map.setView(WORLD_CENTER, WORLD_ZOOM);
        map.closePopup();
        setActivePlace(-1);
      });
    }

    document.querySelectorAll(".travel-map__place").forEach((button) => {
      button.addEventListener("click", function () {
        goToPlace(Number(button.dataset.index), true);
      });
    });

    window.addEventListener("load", function () {
      map.invalidateSize();
    });
    setTimeout(function () {
      map.invalidateSize();
    }, 200);
  }

  function start() {
    if (typeof L === "undefined") {
      window.addEventListener("load", start, { once: true });
      return;
    }
    initTravelMap();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
