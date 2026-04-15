document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const connectBtn = document.getElementById("connect-btn");
  const statusDot = document.querySelector(".status-dot");
  const statusText = document.querySelector(".status-text");
  const videoExpandBtn = document.getElementById("video-expand-btn");
  const loadingOverlay = document.getElementById("loading-overlay");
  const cameraErrorOverlay = document.getElementById("camera-error-overlay");
  const videoFeed = document.getElementById("live-feed");
  const yoloBtn = document.getElementById("yolo-btn");
  const gestureBtn = document.getElementById("gesture-btn");
  const gestureDispatchBtn = document.getElementById("gesture-dispatch-btn");
  const weaponsBtn = document.getElementById("weapons-btn");

  const camName = document.getElementById("cam-name");
  const camFps = document.getElementById("cam-fps");
  const camLatency = document.getElementById("cam-latency");

  const sysCpu = document.getElementById("sys-cpu");
  const sysRam = document.getElementById("sys-ram");
  const sysUptime = document.getElementById("sys-uptime");
  const sysBattery = document.getElementById("sys-battery");
  const sysConnection = document.getElementById("sys-connection");
  const sysTelemetry = document.getElementById("sys-telemetry");
  const sysPeakTemp = document.getElementById("sys-peak-temp");
  const componentList = document.getElementById("component-list");
  const threatList = document.getElementById("threat-list");
  const imuUnitToggle = document.getElementById("imu-unit-toggle");
  const operatorName = document.getElementById("operator-name");
  if (operatorName) {
    operatorName.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); operatorName.blur(); }
    });
  }

  const gestureToast = document.getElementById("gesture-toast");
  const gestureToastLabel = document.getElementById("gesture-toast-label");
  const gestureToastAction = document.getElementById("gesture-toast-action");
  const gestureToastBar = document.getElementById("gesture-toast-bar");

  // State
  let isConnected = false;
  let isYoloEnabled = false;
  let isGestureEnabled = false;
  let isGestureDispatchEnabled = true;
  let isWeaponsEnabled = false;
  let telemetryInterval;
  let currentMode = "go2";
  let imuInDegrees = true;
  let lastImuRpy = null;

  let pc = null;
  let dc = null;
  let pingInterval;
  let lastPingTime = 0;
  let gestureToastTimeout = null;

  const GESTURE_INFO = {
    like:        { label: "THUMBS UP",    action: "STAND UP + WALK" },
    dislike:     { label: "THUMBS DOWN",  action: "STOP + SIT DOWN" },
    peacesign:   { label: "PEACE SIGN",   action: "WAVE HELLO" },
    heart:       { label: "HEART",        action: "HEART POSE" },
    fingerheart: { label: "FINGER HEART", action: "HEART POSE" },
    pinkie:      { label: "PINKIE",       action: "PINKIE GESTURE" },
  };

  const GESTURE_COOLDOWN_MS = 2000;

  // // --- MOCK: remove before prod ---
  // const MOCK_GESTURES = ["like", "dislike", "peacesign", "heart", "fingerheart", "pinkie"];
  // let mockIdx = 0;
  // setTimeout(function fireMock() {
  //   const g = MOCK_GESTURES[mockIdx % MOCK_GESTURES.length];
  //   showGestureToast(g, 0.85 + Math.random() * 0.14);
  //   mockIdx++;
  //   setTimeout(fireMock, 3500);
  // }, 800);
  // // --- END MOCK ---

  function showGestureToast(label, conf) {
    if (!gestureToast) return;
    const info = GESTURE_INFO[label] || { label: label.toUpperCase(), action: "COMMAND SENT" };
    gestureToastLabel.textContent = `${info.label}  ${Math.round(conf * 100)}%`;
    gestureToastAction.textContent = info.action;

    // Reset and animate the progress bar drain
    gestureToastBar.style.transition = "none";
    gestureToastBar.style.transform = "scaleX(1)";
    void gestureToastBar.offsetWidth; // force reflow
    gestureToastBar.style.transition = `transform ${GESTURE_COOLDOWN_MS}ms linear`;
    gestureToastBar.style.transform = "scaleX(0)";

    gestureToast.classList.remove("hidden");
    if (gestureToastTimeout) clearTimeout(gestureToastTimeout);
    gestureToastTimeout = setTimeout(() => {
      gestureToast.classList.add("hidden");
    }, GESTURE_COOLDOWN_MS + 300);
  }

  // Component manifest + session state
  let componentManifest = [];       // [{ id, label, yolo_classes, damaged_classes }]
  let classLookup = {};             // normalised class string → { id, type: "present"|"damaged" }
  let componentState = {};          // id → { status: "missing"|"present"|"damaged", conf }
  let unknownState = {};            // normalised label → { label, conf } — sticky unknowns

  // Threat manifest + session state
  let threatManifest = [];          // [{ id, label, yolo_classes }]
  let threatClassLookup = {};       // normalised class string → { id }
  let threatState = {};             // id → { status: "missing"|"present", conf }

  function normaliseClass(s) {
    return s.toLowerCase().replace(/_/g, " ").trim();
  }

  function buildClassLookup() {
    classLookup = {};
    componentManifest.forEach((comp) => {
      (comp.yolo_classes || []).forEach((cls) => {
        classLookup[normaliseClass(cls)] = { id: comp.id, type: "present" };
      });
      (comp.damaged_classes || []).forEach((cls) => {
        classLookup[normaliseClass(cls)] = { id: comp.id, type: "damaged" };
      });
    });
  }

  function initComponentState() {
    componentState = {};
    componentManifest.forEach((comp) => {
      componentState[comp.id] = { status: "missing", conf: 0 };
    });
    unknownState = {};
    renderComponentList();
  }

  function getStatusIconPath(status) {
    if (status === "missing") return "/static/icons/status-missing.svg";
    if (status === "damaged") return "/static/icons/status-damaged.svg";
    if (status === "present") return "/static/icons/status-present.svg";
    return "";
  }

  function buildCard(label, status, conf) {
    const confStr = conf > 0 ? `CONF: ${Math.round(conf * 100)}%` : "";
    const cssClass =
      status === "present" ? "status-present" :
      status === "damaged" ? "status-warning" :
      status === "unknown" ? "status-unknown" :
      "status-missing";
    const stateLabel =
      status === "present" ? "PRESENT" :
      status === "damaged" ? "DAMAGED" :
      status === "unknown" ? "?" :
      "MISSING";
    const iconPath = getStatusIconPath(status);
    const iconHtml = iconPath
      ? `<img src="${iconPath}" class="status-icon" alt="${stateLabel}" onerror="this.style.display='none'" />`
      : "";
    const newBadge = status === "unknown" ? `<span class="dc-new-badge">NEW</span>` : "";
    return `
      <div class="detection-card ${cssClass}">
        <div class="dc-header">
          <h4>${label}</h4>
          ${iconHtml}
        </div>
        <div class="dc-footer">
          <span class="dc-state">${stateLabel}</span>
          <span style="display:flex;align-items:center;gap:0.4rem;">${newBadge}<span class="dc-conf">${confStr}</span></span>
        </div>
      </div>`;
  }

  function renderComponentList() {
    if (!componentList) return;
    componentList.innerHTML = "";
    componentManifest.forEach((comp) => {
      const state = componentState[comp.id] || { status: "missing", conf: 0 };
      componentList.insertAdjacentHTML("beforeend", buildCard(comp.label, state.status, state.conf));
    });
    Object.values(unknownState).forEach((unk) => {
      componentList.insertAdjacentHTML("beforeend", buildCard(unk.label, "unknown", unk.conf));
    });
  }

  function buildThreatClassLookup() {
    threatClassLookup = {};
    threatManifest.forEach((t) => {
      (t.yolo_classes || []).forEach((cls) => {
        threatClassLookup[normaliseClass(cls)] = { id: t.id };
      });
    });
  }

  function initThreatState() {
    threatState = {};
    threatManifest.forEach((t) => {
      threatState[t.id] = { status: "missing", conf: 0 };
    });
    renderThreatList();
  }

  function renderThreatList() {
    if (!threatList) return;
    threatList.innerHTML = "";
    threatManifest.forEach((t) => {
      const state = threatState[t.id] || { status: "missing", conf: 0 };
      threatList.insertAdjacentHTML("beforeend", buildCard(t.label, state.status, state.conf));
    });
  }

  // Load manifests immediately — cards render as "missing" on load
  fetch("/static/components.json")
    .then((r) => r.json())
    .then((data) => {
      componentManifest = data.components || [];
      buildClassLookup();
      initComponentState();
    })
    .catch(() => {
      componentManifest = [];
      classLookup = {};
    });

  fetch("/static/threats.json")
    .then((r) => r.json())
    .then((data) => {
      threatManifest = data.threats || [];
      buildThreatClassLookup();
      initThreatState();
    })
    .catch(() => {
      threatManifest = [];
      threatClassLookup = {};
    });

  const expandIconSvg = `
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <polyline points="15 3 21 3 21 9"></polyline>
      <line x1="9" y1="21" x2="3" y2="21"></line>
      <line x1="3" y1="21" x2="3" y2="15"></line>
      <line x1="21" y1="3" x2="14" y2="10"></line>
      <line x1="3" y1="21" x2="10" y2="14"></line>
    </svg>
  `;

  const minimizeIconSvg = `
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <line x1="21" y1="3" x2="14" y2="10"></line>
      <polyline points="17 10 14 10 14 7"></polyline>
      <line x1="3" y1="21" x2="10" y2="14"></line>
      <polyline points="7 14 10 14 10 17"></polyline>
    </svg>
  `;

  function getModeFromBadge(badge) {
    const span = badge.querySelector("span");
    const label = span ? span.textContent.trim() : "";

    if (label === "GO2 CAMERA") {
      return "go2";
    }
    if (label === "EXT. CAMERA") {
      return "hd_view";
    }
    if (label === "SENSOR FUSION") {
      return "sensor_fusion";
    }
    return "hd_view";
  }

  function getCamNameForMode(mode) {
    if (mode === "sensor_fusion") {
      return "OAK-D_S2";
    }
    if (mode === "go2") {
      return "GO2_CAMERA";
    }
    return "EXT_CAMERA";
  }

  function updateModeUI(mode) {
    currentMode = mode;
    if (camName) {
      camName.textContent = getCamNameForMode(mode);
    }

    if (gestureDispatchBtn && isConnected) {
      gestureDispatchBtn.style.display = mode === "go2" ? "flex" : "none";
    }
  }

  function getSelectedMode() {
    let selectedMode = "hd_view";
    modeBadges.forEach((badge) => {
      if (badge.classList.contains("active")) {
        selectedMode = getModeFromBadge(badge);
      }
    });
    return selectedMode;
  }

  // Initialize modes
  const modeBadges = document.querySelectorAll(".mode-badge");
  modeBadges.forEach((badge) => {
    badge.addEventListener("click", () => {
      modeBadges.forEach((b) => b.classList.remove("active"));
      badge.classList.add("active");
      updateModeUI(getModeFromBadge(badge));
    });
  });
  updateModeUI(getSelectedMode());

  if (videoExpandBtn) {
    const updateFullscreenIcon = () => {
      videoExpandBtn.innerHTML = document.fullscreenElement ? minimizeIconSvg : expandIconSvg;
    };

    updateFullscreenIcon();

    videoExpandBtn.addEventListener("click", () => {
      const videoContainer = document.querySelector(".video-container");
      if (videoContainer) {
        if (!document.fullscreenElement) {
          if (videoContainer.requestFullscreen) {
            videoContainer.requestFullscreen().catch(err => {
              console.error(`Error attempting to enable fullscreen: ${err.message}`);
            });
          }
          videoContainer.classList.add("fullscreen");
        } else {
          if (document.exitFullscreen) {
            document.exitFullscreen();
          }
        }
      }
    });

    document.addEventListener('fullscreenchange', () => {
      const videoContainer = document.querySelector(".video-container");
      if (!document.fullscreenElement && videoContainer) {
        videoContainer.classList.remove("fullscreen");
      }

      updateFullscreenIcon();
    });
  }

  function setToggleButtonState(button, enabled) {
    if (!button) return;

    if (enabled) {
      button.style.color = "#fff";
      button.style.borderColor = "var(--accent-blue)";
      button.style.background = "var(--accent-blue)";
    } else {
      button.style.color = "";
      button.style.borderColor = "";
      button.style.background = "";
    }
  }

  if (yoloBtn) {
    yoloBtn.addEventListener("click", () => {
      isYoloEnabled = !isYoloEnabled;
      setToggleButtonState(yoloBtn, isYoloEnabled);
      if (dc && dc.readyState === "open") {
        dc.send("toggle_yolo");
      }
    });
  }

  if (gestureBtn) {
    gestureBtn.addEventListener("click", () => {
      isGestureEnabled = !isGestureEnabled;
      setToggleButtonState(gestureBtn, isGestureEnabled);
      if (dc && dc.readyState === "open") {
        dc.send("toggle_gesture");
      }
    });
  }

  if (gestureDispatchBtn) {
    gestureDispatchBtn.addEventListener("click", () => {
      isGestureDispatchEnabled = !isGestureDispatchEnabled;
      setToggleButtonState(gestureDispatchBtn, isGestureDispatchEnabled);
      if (dc && dc.readyState === "open") {
        dc.send("toggle_gesture_dispatch");
      }
    });
  }

  if (weaponsBtn) {
    weaponsBtn.addEventListener("click", () => {
      isWeaponsEnabled = !isWeaponsEnabled;
      setToggleButtonState(weaponsBtn, isWeaponsEnabled);
      if (dc && dc.readyState === "open") {
        dc.send("toggle_weapons");
      }
    });
  }

  // Connect Button Handling
  connectBtn.addEventListener("click", () => {
    if (!isConnected) {
      // Start connection
      connectBtn.classList.add("playing");
      connectBtn.innerHTML =
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"/></svg>';

      loadingOverlay.classList.remove("hidden");
      if (cameraErrorOverlay) cameraErrorOverlay.classList.add("hidden");

      pc = new RTCPeerConnection();
      dc = pc.createDataChannel("telemetry");

      dc.onopen = () => {
        if (currentMode !== "go2") {
          loadingOverlay.classList.add("hidden");
        }
        isConnected = true;

        statusDot.classList.remove("disconnected");
        statusDot.classList.add("connected");
        statusText.classList.remove("disconnected");
        statusText.classList.add("connected");
        statusText.textContent = "LIVE STREAM CONNECTED";

        if (yoloBtn) {
          yoloBtn.style.display = "flex";
          setToggleButtonState(yoloBtn, isYoloEnabled);
        }

        if (gestureBtn) {
          gestureBtn.style.display = "flex";
          setToggleButtonState(gestureBtn, isGestureEnabled);
        }

        if (gestureDispatchBtn) {
          if (currentMode === "go2") {
            gestureDispatchBtn.style.display = "flex";
            setToggleButtonState(gestureDispatchBtn, isGestureDispatchEnabled);
          } else {
            gestureDispatchBtn.style.display = "none";
          }
        }

        if (weaponsBtn) {
          weaponsBtn.style.display = "flex";
          setToggleButtonState(weaponsBtn, isWeaponsEnabled);
        }

        if (isYoloEnabled) {
          dc.send("toggle_yolo");
        }

        if (isGestureEnabled) {
          dc.send("toggle_gesture");
        }

        if (isWeaponsEnabled) {
          dc.send("toggle_weapons");
        }

        startTelemetry();
      };

      dc.onmessage = (event) => {
        if (event.data === "pong") {
          const lat = Date.now() - lastPingTime;
          if (camLatency) { camLatency.textContent = lat; applyLatencyColor(camLatency, lat); }
          return;
        }
        try {
          const data = JSON.parse(event.data);
          if (data.type === "stats") {
            if (data.camera_connected === false) {
              loadingOverlay.classList.add("hidden");
              const _errTitle = document.getElementById("camera-error-text");
              const _errSub   = document.getElementById("camera-error-sub");
              if (_errTitle) {
                const titles = { go2: "GO2 CAMERA UNAVAILABLE", sensor_fusion: "OAK-D UNAVAILABLE", hd_view: "CAMERA UNAVAILABLE" };
                _errTitle.textContent = titles[currentMode] || "CAMERA UNAVAILABLE";
              }
              if (_errSub) {
                const subs = { go2: "CHECK SDK CONNECTION", sensor_fusion: "ENSURE DEPTHAI IS INSTALLED & OAK-D IS CONNECTED", hd_view: "CHECK USB CONNECTION" };
                _errSub.textContent = data.camera_error || subs[currentMode] || "CHECK CONNECTION";
              }
              if (cameraErrorOverlay) cameraErrorOverlay.classList.remove("hidden");
            } else if (data.initializing === false && !loadingOverlay.classList.contains("hidden")) {
              loadingOverlay.classList.add("hidden");
            }
            if (sysCpu) { sysCpu.textContent = `${data.cpu_percent.toFixed(1)}%`; applyThresholdColor(sysCpu, data.cpu_percent, 70, 90); }
            if (sysRam) { sysRam.textContent = `${data.ram_percent.toFixed(1)}%`; applyThresholdColor(sysRam, data.ram_percent, 70, 90); }
            if (sysUptime && data.uptime !== undefined) {
              const hours = Math.floor(data.uptime / 3600);
              const minutes = Math.floor((data.uptime % 3600) / 60);
              const seconds = Math.floor(data.uptime % 60);
              sysUptime.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
            if (componentList) {
              updateDetections([...(data.detections || []), ...(data.weapons_detections || [])]);
            }
            if (camFps && data.fps !== undefined) {
              camFps.textContent = data.fps.toFixed(1);
            }
            if (sysBattery && data.battery !== undefined) {
              sysBattery.textContent = `${data.battery}%`;
            }
            if (sysTelemetry) {
              const avgTemp = Number(data.avg_temp_c);
              if (Number.isFinite(avgTemp)) {
                sysTelemetry.textContent = `${avgTemp.toFixed(1)} °C`;
              } else {
                const temps = Array.isArray(data.motor_temps)
                  ? data.motor_temps
                    .map((value) => Number(value))
                    .filter((value) => Number.isFinite(value))
                  : [];
                if (temps.length > 0) {
                  const fallbackAvgTemp = temps.reduce((sum, value) => sum + value, 0) / temps.length;
                  sysTelemetry.textContent = `${fallbackAvgTemp.toFixed(1)} °C`;
                } else {
                  sysTelemetry.textContent = "-- °C";
                }
              }
            }
            if (sysPeakTemp) {
              const peakTemp = Number(data.peak_temp_c);
              const peakJoint = typeof data.peak_joint_name === "string" && data.peak_joint_name
                ? data.peak_joint_name
                : "--";
              if (Number.isFinite(peakTemp)) {
                sysPeakTemp.textContent = `${peakTemp.toFixed(1)} °C (${peakJoint})`;
              } else {
                const temps = Array.isArray(data.motor_temps)
                  ? data.motor_temps
                    .map((value) => Number(value))
                    .filter((value) => Number.isFinite(value))
                  : [];
                if (temps.length > 0) {
                  const fallbackPeakTemp = Math.max(...temps);
                  sysPeakTemp.textContent = `${fallbackPeakTemp.toFixed(1)} °C (${peakJoint})`;
                } else {
                  sysPeakTemp.textContent = "-- °C (--)";
                }
              }
            }
            if (sysConnection) {
              if (data.connected) {
                sysConnection.textContent = "CONNECTED";
                sysConnection.parentElement.classList.add("accent-green");
                sysConnection.parentElement.classList.remove("accent-red");
              } else {
                sysConnection.textContent = "DISCONNECTED";
                sysConnection.parentElement.classList.remove("accent-green");
                sysConnection.parentElement.classList.add("accent-red");
              }
            }
            if (data.yolo_enabled !== undefined) {
              isYoloEnabled = !!data.yolo_enabled;
              setToggleButtonState(yoloBtn, isYoloEnabled);
            }
            if (data.weapons_enabled !== undefined) {
              isWeaponsEnabled = !!data.weapons_enabled;
              setToggleButtonState(weaponsBtn, isWeaponsEnabled);
            }
            if (data.gesture_enabled !== undefined) {
              isGestureEnabled = !!data.gesture_enabled;
              setToggleButtonState(gestureBtn, isGestureEnabled);
            }
            if (currentMode === "go2" && data.gesture_dispatch_enabled !== undefined) {
              isGestureDispatchEnabled = !!data.gesture_dispatch_enabled;
              setToggleButtonState(gestureDispatchBtn, isGestureDispatchEnabled);
            }
            if (Array.isArray(data.imu_rpy) && data.imu_rpy.length === 3) {
              lastImuRpy = data.imu_rpy;
              renderImuPose();
            }
            if (data.dispatched_gesture) {
              showGestureToast(data.dispatched_gesture.label, data.dispatched_gesture.conf);
            }
          }
        } catch (e) {
          // Ignore non-json
        }
      };

      pc.ontrack = (event) => {
        if (event.track.kind === "video") {
          videoFeed.srcObject = event.streams[0];
        }
      };

      pc.onconnectionstatechange = () => {
        if (pc && (pc.connectionState === "failed" || pc.connectionState === "disconnected" || pc.connectionState === "closed")) {
          doDisconnect();
        }
      };

      // Lock mode badges while connected
      modeBadges.forEach((b) => b.classList.add("locked"));

      // Explicitly request to receive video so the backend knows what to negotiate
      pc.addTransceiver("video", { direction: "recvonly" });

      const selectedMode = getSelectedMode();
      updateModeUI(selectedMode);

      pc.createOffer()
        .then((offer) => pc.setLocalDescription(offer))
        .then(() =>
          fetch("/offer", {
            body: JSON.stringify({
              sdp: pc.localDescription.sdp,
              type: pc.localDescription.type,
              mode: selectedMode,
            }),
            headers: { "Content-Type": "application/json" },
            method: "POST",
          }),
        )
        .then((response) => response.json())
        .then((answer) =>
          pc.setRemoteDescription(new RTCSessionDescription(answer)),
        )
        .catch((err) => {
          console.error("WebRTC Error:", err);
          doDisconnect();
        });
    } else {
      doDisconnect();
    }
  });

  function doDisconnect() {
    if (!isConnected && loadingOverlay.classList.contains("hidden")) return;
    isConnected = false;

    loadingOverlay.classList.add("hidden");
    if (cameraErrorOverlay) cameraErrorOverlay.classList.add("hidden");

    connectBtn.classList.remove("playing");
    connectBtn.innerHTML =
      '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M5 3l14 9-14 9V3z"/></svg>';

    statusDot.classList.add("disconnected");
    statusDot.classList.remove("connected");
    statusText.classList.add("disconnected");
    statusText.classList.remove("connected");
    statusText.textContent = "LIVE STREAM DISCONNECTED";

    if (yoloBtn) yoloBtn.style.display = "none";
    if (gestureBtn) gestureBtn.style.display = "none";
    if (gestureDispatchBtn) gestureDispatchBtn.style.display = "none";
    if (weaponsBtn) weaponsBtn.style.display = "none";

    modeBadges.forEach((b) => b.classList.remove("locked"));

    if (videoFeed.srcObject) {
      videoFeed.srcObject.getTracks().forEach((track) => track.stop());
      videoFeed.srcObject = null;
    }
    if (dc) { dc.close(); dc = null; }
    if (pc) { pc.close(); pc = null; }

    stopTelemetry();
    if (pingInterval) clearInterval(pingInterval);
    initComponentState();
    initThreatState();
  }

  // IMU unit toggle
  function renderImuPose() {
    if (!lastImuRpy) return;
    const poseVals = document.querySelectorAll(".pose-value");
    if (poseVals.length < 3) return;
    const [rawRoll, rawPitch, rawYaw] = lastImuRpy;
    if (imuInDegrees) {
      const R2D = 180 / Math.PI;
      const pitch = rawPitch * R2D;
      const roll  = rawRoll  * R2D;
      const yaw   = rawYaw   * R2D;
      poseVals[0].textContent = (pitch >= 0 ? "+" : "") + pitch.toFixed(1) + "°";
      poseVals[1].textContent = (roll  >= 0 ? "+" : "") + roll.toFixed(1)  + "°";
      poseVals[2].textContent = yaw.toFixed(1) + "°";
    } else {
      poseVals[0].textContent = (rawPitch >= 0 ? "+" : "") + rawPitch.toFixed(3) + " rad";
      poseVals[1].textContent = (rawRoll  >= 0 ? "+" : "") + rawRoll.toFixed(3)  + " rad";
      poseVals[2].textContent = rawYaw.toFixed(3) + " rad";
    }
  }

  // Generate QC Report
  const reportBtn = document.getElementById("btn-report");
  if (reportBtn) {
    reportBtn.addEventListener("click", async () => {
      const operatorEl = document.getElementById("operator-name");
      const operator = prompt("Operator name:", operatorEl ? operatorEl.textContent.trim() : "A. BOCA");
      if (operator === null) return;
      const location = prompt("Location / Zone:", "HANNOVER MESSE");
      if (location === null) return;

      reportBtn.disabled = true;
      reportBtn.textContent = "GENERATING...";

      try {
        const res = await fetch("/report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ operator, location }),
        });

        if (!res.ok) {
          alert("Report generation failed.");
          return;
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const disposition = res.headers.get("Content-Disposition") ?? "";
        const filename = disposition.includes('"')
          ? disposition.split('"')[1]
          : "QC-report.pdf";

        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } catch (err) {
        alert("Report generation failed: " + err.message);
      } finally {
        reportBtn.disabled = false;
        reportBtn.textContent = "GENERATE QC REPORT";
      }
    });
  }

  imuUnitToggle.addEventListener("click", () => {
    imuInDegrees = !imuInDegrees;
    imuUnitToggle.textContent = imuInDegrees ? "DEG" : "RAD";
    renderImuPose();
  });

  // Telemetry
  function startTelemetry() {
    pingInterval = setInterval(() => {
      if (dc && dc.readyState === "open") {
        lastPingTime = Date.now();
        dc.send("ping");
      }
    }, 1000);

  }

  function applyThresholdColor(el, value, warnAt, dangerAt) {
    if (!el) return;
    if (value >= dangerAt) el.style.color = "var(--accent-red)";
    else if (value >= warnAt) el.style.color = "var(--accent-yellow)";
    else el.style.color = "";
  }

function applyLatencyColor(el, ms) {
    if (!el) return;
    if (ms > 150) el.style.color = "var(--accent-red)";
    else if (ms > 50) el.style.color = "var(--accent-yellow)";
    else el.style.color = "var(--accent-green)";
  }

  function stopTelemetry() {
    clearInterval(telemetryInterval);
    if (camFps) camFps.textContent = "--";
    if (camLatency) { camLatency.textContent = "--"; camLatency.style.color = ""; }
    if (sysCpu) { sysCpu.textContent = "--%"; sysCpu.style.color = ""; }
    if (sysRam) { sysRam.textContent = "--%"; sysRam.style.color = ""; }
    if (sysBattery) sysBattery.textContent = "--%";
    if (sysUptime) sysUptime.textContent = "--:--:--";
    if (sysConnection) {
      sysConnection.textContent = "--";
      sysConnection.parentElement.classList.remove("accent-green", "accent-red");
    }
    if (sysTelemetry) sysTelemetry.textContent = "-- °C";
    if (sysPeakTemp) sysPeakTemp.textContent = "-- °C (--)";
    lastImuRpy = null;
    const poseVals = document.querySelectorAll(".pose-value");
    if (poseVals.length >= 3) {
      poseVals[0].textContent = "--";
      poseVals[1].textContent = "--";
      poseVals[2].textContent = "--";
    }
  }

  function updateDetections(detections) {
    detections.forEach((d) => {
      if (!d.class) return;
      const norm = normaliseClass(d.class);
      const compMatch = classLookup[norm];
      const threatMatch = threatClassLookup[norm];

      if (compMatch) {
        const cur = componentState[compMatch.id];
        if (compMatch.type === "damaged") {
          componentState[compMatch.id] = { status: "damaged", conf: d.conf };
        } else if (cur && cur.status !== "damaged") {
          componentState[compMatch.id] = { status: "present", conf: d.conf };
        }
      } else if (threatMatch) {
        const cur = threatState[threatMatch.id];
        if (cur && cur.status !== "present") {
          threatState[threatMatch.id] = { status: "present", conf: d.conf };
        } else if (cur && d.conf > cur.conf) {
          threatState[threatMatch.id] = { status: "present", conf: d.conf };
        }
      } else {
        // Not in any manifest — sticky unknown for component list
        const readable = d.class.replace(/_/g, " ");
        const label = readable.charAt(0).toUpperCase() + readable.slice(1);
        if (!unknownState[norm] || d.conf > unknownState[norm].conf) {
          unknownState[norm] = { label, conf: d.conf };
        }
      }
    });

    renderComponentList();
    renderThreatList();
  }
});
