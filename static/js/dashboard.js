document.addEventListener("DOMContentLoaded", function () {

    // =====================================================
    // SECTION NAVIGATION
    // =====================================================

    const navButtons = document.querySelectorAll("[data-section]");
    const sections = document.querySelectorAll(".dashboard-section");

    function showSection(sectionName) {

        sections.forEach(function (section) {
            section.classList.remove("active");
        });

        const target = document.getElementById(sectionName);

        if (target) {
            target.classList.add("active");
        }

        navButtons.forEach(function (button) {
            button.classList.remove("active");

            if (button.dataset.section === sectionName) {
                button.classList.add("active");
            }
        });

        if (sectionName === "graphs") {
            setTimeout(function () {
                window.dispatchEvent(new Event("resize"));
                if (typeof initAnalyticsCharts === "function") {
                    initAnalyticsCharts();
                }
            }, 60);
        }
    }

    navButtons.forEach(function (button) {

        button.addEventListener("click", function (event) {

            event.preventDefault();

            showSection(button.dataset.section);

        });

    });


    // =====================================================
    // START PREDICTION BUTTON
    // =====================================================

    const startButtons = document.querySelectorAll(
        '[data-section="prediction"]'
    );

    startButtons.forEach(function (button) {

        button.addEventListener("click", function (event) {

            event.preventDefault();

            showSection("prediction");

            // Scroll to prediction area
            const predictionSection =
                document.getElementById("prediction");

            if (predictionSection) {
                predictionSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }

        });

    });


    // =====================================================
    // PREDICTION FORM
    // =====================================================

    const predictionForm =
        document.querySelector("form");

    if (predictionForm) {

        predictionForm.addEventListener(
            "submit",
            function () {

                const submitButton =
                    predictionForm.querySelector(
                        'button[type="submit"]'
                    );

                if (submitButton) {

                    submitButton.disabled = true;

                    submitButton.innerHTML =
                        "Analyzing accident...";
                }

            }
        );
    }


    // =====================================================
    // PROBABILITY CHART
    // =====================================================

    const chartCanvas =
        document.getElementById("severityChart");

    if (
        chartCanvas &&
        typeof Chart !== "undefined"
    ) {

        const minor =
            parseFloat(
                chartCanvas.dataset.minor || "0"
            );

        const serious =
            parseFloat(
                chartCanvas.dataset.serious || "0"
            );

        const fatal =
            parseFloat(
                chartCanvas.dataset.fatal || "0"
            );

        new Chart(chartCanvas, {

            type: "doughnut",

            data: {

                labels: [
                    "Minor",
                    "Serious",
                    "Fatal"
                ],

                datasets: [{
                    data: [
                        minor,
                        serious,
                        fatal
                    ]
                }]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        position: "bottom"
                    }

                }
            }

        });
    }


    // =====================================================
    // 4-CHART VISUAL ANALYTICS GALLERY (Grey + Black + White UI)
    // =====================================================

    let createdCharts = {};

    function initAnalyticsCharts() {
        if (typeof Chart === "undefined") return;

        const distCanvas = document.getElementById("loadDistributionChart");
        const accCanvas = document.getElementById("accuracyChart");
        const featCanvas = document.getElementById("featureImportanceChart");
        const weatherCanvas = document.getElementById("weatherSeverityChart");

        if (!distCanvas && !accCanvas && !featCanvas && !weatherCanvas) return;

        // If already rendered, just resize
        if (Object.keys(createdCharts).length > 0) {
            Object.values(createdCharts).forEach(function (chart) {
                if (chart && typeof chart.resize === "function") {
                    chart.resize();
                }
            });
            return;
        }

        let analyticsData = null;
        const scriptData = document.getElementById("analyticsData");
        if (scriptData && scriptData.textContent.trim()) {
            try {
                analyticsData = JSON.parse(scriptData.textContent);
            } catch (e) {
                console.warn("Could not parse analytics data from script tag:", e);
            }
        }

        function renderCharts(data) {
            if (!data) return;

            // 1. DISTRIBUTION BAR CHART
            if (distCanvas && !createdCharts.dist) {
                const distData = data.distribution || {
                    labels: ["0% - 20%", "20% - 40%", "40% - 60%", "60% - 80%", "80% - 100%"],
                    counts: [175, 754, 1142, 768, 161]
                };

                createdCharts.dist = new Chart(distCanvas, {
                    type: "bar",
                    data: {
                        labels: distData.labels,
                        datasets: [{
                            label: "Record Count",
                            data: distData.counts,
                            backgroundColor: "#737f90",
                            hoverBackgroundColor: "#8d99aa",
                            borderRadius: 3,
                            borderSkipped: false,
                            barPercentage: 0.72,
                            categoryPercentage: 0.88
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: "#151c28",
                                titleColor: "#ffffff",
                                bodyColor: "#cbd5e1",
                                borderColor: "#2b374e",
                                borderWidth: 1,
                                padding: 10,
                                cornerRadius: 6,
                                displayColors: false,
                                callbacks: {
                                    label: function (ctx) {
                                        return "Record Count: " + ctx.parsed.y.toLocaleString();
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                grid: { display: false, drawBorder: false },
                                ticks: {
                                    color: "#8893a4",
                                    font: { size: 11, family: "'DM Sans', sans-serif" }
                                }
                            },
                            y: {
                                beginAtZero: true,
                                grid: { color: "rgba(255, 255, 255, 0.04)", drawBorder: false },
                                ticks: {
                                    color: "#8893a4",
                                    font: { size: 11, family: "'DM Sans', sans-serif" },
                                    callback: function (val) { return val.toLocaleString(); }
                                }
                            }
                        }
                    }
                });
            }

            // 2. MODEL ACCURACY ACTUAL VS PREDICTED LINE CHART
            if (accCanvas && !createdCharts.acc) {
                const accData = data.actual_vs_pred || { labels: [], actual: [], predicted: [] };

                createdCharts.acc = new Chart(accCanvas, {
                    type: "line",
                    data: {
                        labels: accData.labels,
                        datasets: [
                            {
                                label: "Actual Severity",
                                data: accData.actual,
                                borderColor: "#ffffff",
                                borderWidth: 2,
                                backgroundColor: "#ffffff",
                                pointBackgroundColor: "#ffffff",
                                pointBorderColor: "#0e131e",
                                pointBorderWidth: 1.5,
                                pointRadius: 3.5,
                                pointHoverRadius: 5.5,
                                tension: 0.15,
                                fill: false
                            },
                            {
                                label: "Predicted Severity",
                                data: accData.predicted,
                                borderColor: "#737f90",
                                borderWidth: 2,
                                borderDash: [5, 5],
                                backgroundColor: "#737f90",
                                pointBackgroundColor: "#737f90",
                                pointBorderColor: "#0e131e",
                                pointBorderWidth: 1,
                                pointRadius: 2.5,
                                pointHoverRadius: 4.5,
                                tension: 0.15,
                                fill: false
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { mode: "index", intersect: false },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: "#151c28",
                                titleColor: "#ffffff",
                                bodyColor: "#cbd5e1",
                                borderColor: "#2b374e",
                                borderWidth: 1,
                                padding: 10,
                                cornerRadius: 6,
                                callbacks: {
                                    label: function (ctx) {
                                        return " " + ctx.dataset.label + ": " + ctx.parsed.y.toLocaleString();
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                grid: { color: "rgba(255, 255, 255, 0.03)", drawBorder: false },
                                ticks: {
                                    color: "#8893a4",
                                    font: { size: 11, family: "'DM Sans', sans-serif" },
                                    autoSkip: false,
                                    callback: function (val, index) {
                                        return (index % 3 === 0 || index === accData.labels.length - 1)
                                            ? this.getLabelForValue(val)
                                            : "";
                                    }
                                }
                            },
                            y: {
                                beginAtZero: true,
                                grid: { color: "rgba(255, 255, 255, 0.04)", drawBorder: false },
                                ticks: {
                                    color: "#8893a4",
                                    stepSize: 500,
                                    font: { size: 11, family: "'DM Sans', sans-serif" },
                                    callback: function (val) { return val.toLocaleString(); }
                                }
                            }
                        }
                    }
                });
            }

            // 3. FEATURE IMPORTANCE HORIZONTAL BAR CHART
            if (featCanvas && !createdCharts.feat) {
                const featData = data.feature_importance || {
                    labels: ["Speed Limit", "Driver Age", "Casualties", "Fatalities", "Year", "Vehicles Involved", "Traffic Signs", "Alcohol"],
                    importance: [4.22, 3.96, 3.32, 2.86, 2.79, 2.54, 1.22, 1.19]
                };

                createdCharts.feat = new Chart(featCanvas, {
                    type: "bar",
                    data: {
                        labels: featData.labels,
                        datasets: [{
                            label: "Relative Importance (%)",
                            data: featData.importance,
                            backgroundColor: "#cbd5e1",
                            hoverBackgroundColor: "#ffffff",
                            borderRadius: 3,
                            barPercentage: 0.7
                        }]
                    },
                    options: {
                        indexAxis: "y",
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: "#151c28",
                                titleColor: "#ffffff",
                                bodyColor: "#cbd5e1",
                                borderColor: "#2b374e",
                                borderWidth: 1,
                                padding: 10,
                                cornerRadius: 6,
                                displayColors: false,
                                callbacks: {
                                    label: function (ctx) {
                                        return "Contribution: " + ctx.parsed.x + "%";
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                beginAtZero: true,
                                grid: { color: "rgba(255, 255, 255, 0.04)", drawBorder: false },
                                ticks: {
                                    color: "#8893a4",
                                    font: { size: 11, family: "'DM Sans', sans-serif" },
                                    callback: function (val) { return val + "%"; }
                                }
                            },
                            y: {
                                grid: { display: false, drawBorder: false },
                                ticks: {
                                    color: "#cbd5e1",
                                    font: { size: 11, family: "'DM Sans', sans-serif" }
                                }
                            }
                        }
                    }
                });
            }

            // 4. WEATHER & SEVERITY GROUPED BAR CHART
            if (weatherCanvas && !createdCharts.weather) {
                const wData = data.weather_severity || {
                    conditions: ["Clear", "Rainy", "Foggy", "Stormy", "Hazy"],
                    minor: [202, 224, 174, 231, 203],
                    serious: [182, 210, 207, 177, 205],
                    fatal: [190, 197, 195, 203, 200]
                };

                createdCharts.weather = new Chart(weatherCanvas, {
                    type: "bar",
                    data: {
                        labels: wData.conditions,
                        datasets: [
                            {
                                label: "Minor",
                                data: wData.minor,
                                backgroundColor: "#64748b",
                                hoverBackgroundColor: "#78879e",
                                borderRadius: 3
                            },
                            {
                                label: "Serious",
                                data: wData.serious,
                                backgroundColor: "#94a3b8",
                                hoverBackgroundColor: "#a8b6cb",
                                borderRadius: 3
                            },
                            {
                                label: "Fatal",
                                data: wData.fatal,
                                backgroundColor: "#ffffff",
                                hoverBackgroundColor: "#f1f5f9",
                                borderRadius: 3
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: "#151c28",
                                titleColor: "#ffffff",
                                bodyColor: "#cbd5e1",
                                borderColor: "#2b374e",
                                borderWidth: 1,
                                padding: 10,
                                cornerRadius: 6,
                                callbacks: {
                                    label: function (ctx) {
                                        return " " + ctx.dataset.label + ": " + ctx.parsed.y + " incidents";
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                grid: { display: false, drawBorder: false },
                                ticks: {
                                    color: "#8893a4",
                                    font: { size: 11, family: "'DM Sans', sans-serif" }
                                }
                            },
                            y: {
                                beginAtZero: true,
                                grid: { color: "rgba(255, 255, 255, 0.04)", drawBorder: false },
                                ticks: {
                                    color: "#8893a4",
                                    font: { size: 11, family: "'DM Sans', sans-serif" }
                                }
                            }
                        }
                    }
                });
            }
        }

        if (analyticsData && analyticsData.distribution) {
            renderCharts(analyticsData);
        } else {
            fetch("/api/analytics")
                .then(function (res) { return res.json(); })
                .then(function (res) {
                    if (res.status === "success" && res.data) {
                        renderCharts(res.data);
                    }
                })
                .catch(function (err) {
                    console.warn("Error fetching /api/analytics:", err);
                });
        }
    }

    // Graph Selector / Filter Buttons
    const filterBtns = document.querySelectorAll(".graph-tab-btn");
    const galleryCards = document.querySelectorAll("#graphsGalleryGrid .analytics-card");

    filterBtns.forEach(function (btn) {
        btn.addEventListener("click", function () {
            filterBtns.forEach(function (b) { b.classList.remove("active"); });
            btn.classList.add("active");

            const target = btn.dataset.graphTarget;
            galleryCards.forEach(function (card) {
                if (target === "all") {
                    card.classList.remove("is-hidden");
                    card.classList.remove("is-focused");
                } else {
                    if (card.id === target) {
                        card.classList.remove("is-hidden");
                        card.classList.add("is-focused");
                    } else {
                        card.classList.add("is-hidden");
                        card.classList.remove("is-focused");
                    }
                }
            });

            setTimeout(function () {
                window.dispatchEvent(new Event("resize"));
            }, 60);
        });
    });

    initAnalyticsCharts();


    // =====================================================
    // AFTER A PREDICTION
    // =====================================================
    //
    // If Flask has returned a prediction, automatically
    // show the overview/dashboard instead of leaving the
    // user on the empty welcome screen.
    //
    // =====================================================

    const predictionResult =
        document.querySelector(
            "[data-prediction-result]"
        );

    if (predictionResult) {

        showSection("overview");

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });

    }

});