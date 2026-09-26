#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
root <- normalizePath(file.path(dirname(script_path), ".."))
.libPaths(c(file.path(root, ".r-lib"), .libPaths()))

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(jsonlite)
  library(dplyr)
  library(svglite)
  library(ragg)
})

# Figure contract
# Core conclusion: ECON delays sustained main-engine firing by about 0.32 s,
# consistently across nine matched initial angles.
# Archetype: quantitative grid; panel a is primary evidence, panel b is robustness.
# Output: 183 x 92 mm, editable SVG/PDF plus 600-dpi TIFF and 300-dpi PNG.

study <- file.path(root, "outputs", "lunar_lander_braking_formal_v3_persistent_onset")
summary_path <- file.path(
  study, "v7_angle_grid_extension", "aggregate", "summary.json"
)
episode_root <- file.path(study, "held_out_eval_amended_original_only")
output_dir <- file.path(root, "outputs", "figures")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(summary_path)) stop("Missing nine-angle aggregate summary: ", summary_path)

palette <- c(
  DESC = "#2B7A8B",
  BAL = "#7765A5",
  ECON = "#C8772B",
  angle = "#D79A62",
  combined = "#174A63",
  neutral = "#6D7882",
  pale = "#DCE4E8"
)

theme_pub <- function(base_size = 7.2) {
  theme_classic(base_size = base_size, base_family = "Arial") +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "#26323A"),
      axis.ticks = element_line(linewidth = 0.35, colour = "#26323A"),
      axis.text = element_text(colour = "#26323A"),
      axis.title = element_text(colour = "#17212B"),
      plot.title = element_text(size = base_size + 0.8, face = "bold", colour = "#17212B"),
      plot.subtitle = element_text(size = base_size - 0.4, colour = "#596873", margin = margin(b = 5)),
      plot.caption = element_text(size = base_size - 1, colour = "#596873", hjust = 0),
      panel.grid.major.x = element_line(linewidth = 0.25, colour = "#E2E8EB"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(5, 7, 5, 5),
      legend.position = "none"
    )
}

extract_number <- function(lines, field) {
  pattern <- paste0('.*"', field, '": ([^,}]+).*')
  value <- sub(pattern, "\\1", lines, perl = TRUE)
  value[value == "null"] <- NA_character_
  as.numeric(value)
}

# Panel a source: original intervention, one median per training seed and condition.
episode_files <- Sys.glob(file.path(episode_root, "*", "seed_*", "episodes.jsonl"))
if (length(episode_files) != 60) stop("Expected 60 original held-out episode files")

seed_rows <- lapply(episode_files, function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[grepl('"intervention": "original"', lines, fixed = TRUE)]
  if (length(lines) != 18) stop("Expected 18 original episodes in ", path)
  condition <- basename(dirname(dirname(path)))
  seed <- as.integer(sub("seed_", "", basename(dirname(path)), fixed = TRUE))
  data.frame(
    condition = condition,
    seed = seed,
    onset_s = extract_number(lines, "primary_onset_time_s")
  )
})

panel_a_source <- bind_rows(seed_rows) |>
  group_by(condition, seed) |>
  summarise(onset_s = median(onset_s), .groups = "drop") |>
  mutate(condition = factor(condition, levels = c("ECON", "BAL", "DESC")))

if (nrow(panel_a_source) != 60 || anyNA(panel_a_source$onset_s)) {
  stop("Panel a source must contain 60 observed seed medians")
}

panel_a_medians <- panel_a_source |>
  group_by(condition) |>
  summarise(median_s = median(onset_s), .groups = "drop")

# Panel b source: angle-specific and pooled seed-bootstrap contrasts.
summary <- fromJSON(summary_path, simplifyVector = FALSE)
by_angle <- summary$contrasts$original$by_angle
angle_rows <- lapply(-4:4, function(angle) {
  value <- by_angle[[as.character(angle)]]
  data.frame(
    label = sprintf("%+d°", angle),
    y = 6 - angle,
    estimate_s = value$estimate_s,
    lower_s = value$ci95_s[[1]],
    upper_s = value$ci95_s[[2]],
    type = "Angle-specific"
  )
})

combined_original <- summary$contrasts$original$combined
combined_low <- summary$contrasts$low_descent_speed$combined
panel_b_source <- bind_rows(
  bind_rows(angle_rows),
  data.frame(
    label = "All angles",
    y = 0.55,
    estimate_s = combined_original$estimate_s,
    lower_s = combined_original$ci95_s[[1]],
    upper_s = combined_original$ci95_s[[2]],
    type = "Combined original"
  ),
  data.frame(
    label = "All angles, ½ speed",
    y = -0.65,
    estimate_s = combined_low$estimate_s,
    lower_s = combined_low$ci95_s[[1]],
    upper_s = combined_low$ci95_s[[2]],
    type = "Combined half-speed"
  )
)

write.csv(
  transform(panel_a_source, condition = as.character(condition)),
  file.path(output_dir, "lunar_v7_onset_panel_a_source.csv"),
  row.names = FALSE
)
write.csv(
  panel_b_source,
  file.path(output_dir, "lunar_v7_onset_panel_b_source.csv"),
  row.names = FALSE
)

p_a <- ggplot(panel_a_source, aes(x = onset_s, y = condition, colour = condition)) +
  geom_point(
    position = position_jitter(width = 0, height = 0.075, seed = 260916),
    size = 1.65, alpha = 0.62, stroke = 0
  ) +
  geom_segment(
    data = panel_a_medians,
    aes(x = median_s, xend = median_s, y = as.numeric(condition) - 0.19,
        yend = as.numeric(condition) + 0.19, colour = condition),
    linewidth = 2.2, lineend = "round", inherit.aes = FALSE
  ) +
  geom_text(
    data = panel_a_medians,
    aes(x = 0.665, y = condition, label = sprintf("%.2f s", median_s)),
    inherit.aes = FALSE, hjust = 1, size = 2.55, family = "Arial", colour = "#26323A"
  ) +
  scale_colour_manual(values = palette[c("DESC", "BAL", "ECON")]) +
  scale_x_continuous(
    limits = c(0, 0.68), breaks = seq(0, 0.6, 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  labs(
    title = "Original held-out (initial angle 0°)",
    subtitle = "Each dot: one seed median\nacross 18 matched scenes",
    x = "Sustained-firing onset after matched start (s)", y = NULL
  ) +
  theme_pub() +
  theme(
    axis.text.y = element_text(size = 7.5, colour = "#17212B"),
    panel.grid.major.x = element_line(linewidth = 0.3, colour = "#DFE6E9")
  )

angle_data <- panel_b_source[panel_b_source$type == "Angle-specific", ]
combined_data <- panel_b_source[panel_b_source$type != "Angle-specific", ]

p_b <- ggplot() +
  geom_vline(xintercept = 0, linewidth = 0.45, colour = "#9BA7AE") +
  geom_segment(
    data = angle_data,
    aes(x = lower_s, xend = upper_s, y = y, yend = y),
    linewidth = 0.75, colour = palette[["angle"]], lineend = "round"
  ) +
  geom_point(
    data = angle_data, aes(x = estimate_s, y = y),
    size = 2.15, shape = 21, fill = palette[["angle"]], colour = "white", stroke = 0.45
  ) +
  geom_segment(
    data = combined_data,
    aes(x = lower_s, xend = upper_s, y = y, yend = y, colour = type),
    linewidth = 1.05, lineend = "round"
  ) +
  geom_point(
    data = combined_data,
    aes(x = estimate_s, y = y, fill = type, shape = type),
    size = 2.55, colour = "white", stroke = 0.5
  ) +
  geom_hline(yintercept = 1.3, linewidth = 0.35, colour = palette[["pale"]]) +
  geom_text(
    data = combined_data,
    aes(x = upper_s + 0.012, y = y, label = sprintf("%+.3f", estimate_s), colour = type),
    hjust = 0, size = 2.35, family = "Arial", fontface = "bold"
  ) +
  scale_colour_manual(values = c(
    "Combined original" = palette[["combined"]],
    "Combined half-speed" = palette[["neutral"]]
  )) +
  scale_fill_manual(values = c(
    "Combined original" = palette[["combined"]],
    "Combined half-speed" = "white"
  )) +
  scale_shape_manual(values = c("Combined original" = 21, "Combined half-speed" = 23)) +
  scale_x_continuous(
    limits = c(0, 0.41), breaks = seq(0, 0.4, 0.1),
    expand = expansion(mult = c(0, 0))
  ) +
  scale_y_continuous(
    breaks = c(10:2, 0.55, -0.65),
    labels = c(sprintf("%+d°", -4:4), "All angles", "All angles, ½ speed")
  ) +
  labs(
    title = "Angle-grid robustness",
    subtitle = "ECON − DESC; medians across 20 training seeds\n95% whole-seed bootstrap CI (18 scenes per angle)",
    x = "Difference in sustained-firing onset (s)  →  ECON later", y = NULL
  ) +
  coord_cartesian(clip = "off") +
  theme_pub() +
  theme(
    axis.line.y = element_blank(), axis.ticks.y = element_blank(),
    axis.text.y = element_text(size = 6.7),
    panel.grid.major.x = element_line(linewidth = 0.3, colour = "#DFE6E9")
  )

figure <- p_a + p_b +
  plot_layout(widths = c(1.03, 1)) +
  plot_annotation(
    title = "LunarLander | sustained main-engine firing",
    subtitle = paste0(
      "v7 amended primary result with post-result nine-angle robustness extension",
      "   ·   primary: +0.31 s   ·   nine-angle pooled: +0.32 s"
    ),
    caption = paste0(
      "Primary panel: original 0° held-out scenes. Robustness panel: exploratory angle extension. ",
      "Intervals are descriptive; n = 20 training seeds."
    ),
    tag_levels = "a",
    theme = theme(
      text = element_text(family = "Arial", colour = "#17212B"),
      plot.title = element_text(size = 10.5, face = "bold", margin = margin(b = 2)),
      plot.subtitle = element_text(size = 7.2, colour = "#596873", margin = margin(b = 5)),
      plot.caption = element_text(size = 6.1, colour = "#596873", hjust = 0, margin = margin(t = 5)),
      plot.tag = element_text(size = 8.5, face = "bold", colour = "#17212B")
    )
  )

base <- file.path(output_dir, "lunar_v7_onset_angle_grid")
width_in <- 183 / 25.4
height_in <- 92 / 25.4

svglite(paste0(base, ".svg"), width = width_in, height = height_in)
print(figure)
dev.off()

cairo_pdf(paste0(base, ".pdf"), width = width_in, height = height_in, family = "Arial")
print(figure)
dev.off()

agg_tiff(
  paste0(base, ".tiff"), width = width_in, height = height_in,
  units = "in", res = 600, background = "white"
)
print(figure)
dev.off()

agg_png(
  paste0(base, ".png"), width = width_in, height = height_in,
  units = "in", res = 300, background = "white"
)
print(figure)
dev.off()

cat(paste0(base, c(".svg", ".pdf", ".tiff", ".png")), sep = "\n")
