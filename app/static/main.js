document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const connectBtn = document.getElementById("connect-btn");
  const statusDot = document.querySelector(".status-dot");
  const statusText = document.querySelector(".status-text");
  const videoExpandBtn = document.getElementById("video-expand-btn");
  const loadingOverlay = document.getElementById("loading-overlay");
  const videoFeed = document.getElementById("live-feed");
  const yoloBtn = document.getElementById("yolo-btn");
  const gestureBtn = document.getElementById("gesture-btn");
  const gestureDispatchBtn = document.getElementById("gesture-dispatch-btn");

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

  // State
  let isConnected = false;
  let isYoloEnabled = false;
  let isGestureEnabled = false;
  let isGestureDispatchEnabled = true;
  let telemetryInterval;
  let currentMode = "hd_view";

  let pc = null;
  let dc = null;
  let pingInterval;
  let lastPingTime = 0;

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
    if (label === "THERMAL") {
      return "thermal";
    }
    if (label === "SENSOR FUSION") {
      return "sensor_fusion";
    }
    return "hd_view";
  }

  function getCamNameForMode(mode) {
    if (mode === "thermal") {
      return "THERMAL_CAMERA";
    }
    if (mode === "sensor_fusion") {
      return "SENSOR_FUSION_CAM";
    }
    if (mode === "go2") {
      return "GO2_CAMERA";
    }
    return "GENERIC_USB_CAM";
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

  // Connect Button Handling
  connectBtn.addEventListener("click", () => {
    if (!isConnected) {
      // Start connection
      connectBtn.classList.add("playing");
      connectBtn.innerHTML =
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12"/></svg>';

      loadingOverlay.classList.remove("hidden");

      pc = new RTCPeerConnection();
      dc = pc.createDataChannel("telemetry");

      dc.onopen = () => {
        loadingOverlay.classList.add("hidden");
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

        if (isYoloEnabled) {
          dc.send("toggle_yolo");
        }

        if (isGestureEnabled) {
          dc.send("toggle_gesture");
        }

        startTelemetry();
      };

      dc.onmessage = (event) => {
        if (event.data === "pong") {
          const lat = Date.now() - lastPingTime;
          if (camLatency) camLatency.textContent = lat;
          return;
        }
        try {
          const data = JSON.parse(event.data);
          if (data.type === "stats") {
            if (sysCpu) sysCpu.textContent = `${data.cpu_percent.toFixed(1)}%`;
            if (sysRam) sysRam.textContent = `${data.ram_percent.toFixed(1)}%`;
            if (sysUptime && data.uptime !== undefined) {
              const hours = Math.floor(data.uptime / 3600);
              const minutes = Math.floor((data.uptime % 3600) / 60);
              const seconds = Math.floor(data.uptime % 60);
              sysUptime.textContent = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
            if (componentList) {
              const merged = [
                ...(data.detections || []),
                ...(data.gestures || []),
              ];
              updateDetections(merged);
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
            if (currentMode === "go2" && data.gesture_dispatch_enabled !== undefined) {
              isGestureDispatchEnabled = !!data.gesture_dispatch_enabled;
              setToggleButtonState(gestureDispatchBtn, isGestureDispatchEnabled);
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
          loadingOverlay.classList.add("hidden");
          // Reset UI on failure
          connectBtn.classList.remove("playing");
          connectBtn.innerHTML =
            '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M5 3l14 9-14 9V3z"/></svg>';
        });
    } else {
      // Disconnect
      isConnected = false;
      connectBtn.classList.remove("playing");
      connectBtn.innerHTML =
        '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M5 3l14 9-14 9V3z"/></svg>';

      // Update status UI
      statusDot.classList.add("disconnected");
      statusDot.classList.remove("connected");
      statusText.classList.add("disconnected");
      statusText.classList.remove("connected");
      statusText.textContent = "LIVE STREAM DISCONNECTED";

      if (yoloBtn) {
        yoloBtn.style.display = "none";
      }

      if (gestureBtn) {
        gestureBtn.style.display = "none";
      }

      if (gestureDispatchBtn) {
        gestureDispatchBtn.style.display = "none";
      }

      if (videoFeed.srcObject) {
        videoFeed.srcObject.getTracks().forEach((track) => track.stop());
        videoFeed.srcObject = null;
      }
      if (dc) {
        dc.close();
        dc = null;
      }
      if (pc) {
        pc.close();
        pc = null;
      }

      stopTelemetry();
      if (pingInterval) {
        clearInterval(pingInterval);
      }
    }
  });

  // Telemetry
  function startTelemetry() {
    pingInterval = setInterval(() => {
      if (dc && dc.readyState === "open") {
        lastPingTime = Date.now();
        dc.send("ping");
      }
    }, 1000);

    telemetryInterval = setInterval(() => {
      // Simulate pose fluctuations (these are still simulated for now)
      const pitch = (2.4 + (Math.random() - 0.5) * 0.2).toFixed(1);
      const roll = (-0.1 + (Math.random() - 0.5) * 0.1).toFixed(1);
      const yaw = (184.2 + (Math.random() - 0.5) * 0.5).toFixed(1);

      const poseVals = document.querySelectorAll(".pose-value");
      if (poseVals.length >= 3) {
        poseVals[0].textContent = (pitch >= 0 ? "+" : "") + pitch + "°";
        poseVals[1].textContent = (roll >= 0 ? "+" : "") + roll + "°";
        poseVals[2].textContent = yaw + "°";
      }
    }, 1000);
  }

  function stopTelemetry() {
    clearInterval(telemetryInterval);
    camFps.textContent = "--";
    camLatency.textContent = "--";
    if (sysTelemetry) {
      sysTelemetry.textContent = "-- °C";
    }
    if (sysPeakTemp) {
      sysPeakTemp.textContent = "-- °C (--)";
    }

    // Reset pose values
    const poseVals = document.querySelectorAll(".pose-value");
    if (poseVals.length >= 3) {
      poseVals[0].textContent = "+2.4°";
      poseVals[1].textContent = "-0.1°";
      poseVals[2].textContent = "184.2°";
    }
  }

  function getStatusIconPath(status) {
    if (status === "missing") {
      return "/static/icons/status-missing.svg";
    }
    if (status === "damaged") {
      return "/static/icons/status-damaged.svg";
    }
    return "/static/icons/status-present.svg";
  }

  function updateDetections(detections) {
    if (!componentList) return;

    // Group detections by class name and get the highest confidence
    const uniqueDets = {};
    detections.forEach((d) => {
      if (!d.class) return;
      // Capitalize the class name nicely
      const readableName = d.class
        .replace("gesture:", "Gesture ")
        .replace(/_/g, " ");
      const clsName = readableName.charAt(0).toUpperCase() + readableName.slice(1);
      if (!uniqueDets[clsName] || uniqueDets[clsName] < d.conf) {
        uniqueDets[clsName] = d.conf;
      }
    });

    // Update existing cards or create new ones
    Object.keys(uniqueDets).forEach((clsName) => {
      let confPercent = Math.round(uniqueDets[clsName] * 100);

      let existingCards = Array.from(componentList.querySelectorAll('.detection-card'));
      let foundCard = existingCards.find((card) => {
        let title = card.querySelector('h4');
        return title && title.textContent === clsName;
      });

      if (foundCard) {
        // Update the card if it exists (handles both hardcoded and dynamic)
        foundCard.className = 'detection-card status-present';
        foundCard.querySelector('.dc-state').textContent = 'PRESENT';
        foundCard.querySelector('.dc-conf').textContent = `CONF: ${confPercent}%`;
        const statusIcon = foundCard.querySelector('.status-icon');
        if (statusIcon) {
          statusIcon.src = getStatusIconPath('present');
          statusIcon.alt = 'Present';
        }
      } else {
        // Create a new dynamic card specifically for computer vision targets
        const cardHTML = `
            <div class="detection-card status-present" data-dynamic="true">
              <div class="dc-header">
                <h4>${clsName}</h4>
                <img src="${getStatusIconPath('present')}" class="status-icon" alt="Present"
                  onerror="this.style.display='none'" />
              </div>
              <div class="dc-footer">
                <span class="dc-state">PRESENT</span>
                <span class="dc-conf">CONF: ${confPercent}%</span>
              </div>
            </div>
        `;
        // Insert at the top of the list so it's easily noticeable
        componentList.insertAdjacentHTML('afterbegin', cardHTML);
      }
    });

    // Set dynamic cards to missing if no longer detected by YOLO (doesn't wipe hardcoded ones)
    let dynamicCards = Array.from(componentList.querySelectorAll('.detection-card[data-dynamic="true"]'));
    dynamicCards.forEach((card) => {
      let title = card.querySelector('h4').textContent;
      if (!uniqueDets[title]) {
        card.className = 'detection-card status-missing';
        card.querySelector('.dc-state').textContent = 'MISSING';
        const statusIcon = card.querySelector('.status-icon');
        if (statusIcon) {
          statusIcon.src = getStatusIconPath('missing');
          statusIcon.alt = 'Missing';
        }
      }
    });
  }
});
