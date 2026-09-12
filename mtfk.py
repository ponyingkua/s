        pad = y_span * 0.12
        ax_p.set_ylim(y_low - pad, y_high + pad)
        ax_p.set_xlim(-0.6, last_x + 0.6)
        ax_v.set_xlim(-0.6, last_x + 0.6)

        vol_lookback = cfg.get("indicators", {}).get("volume_spike", {}).get("lookback", 20)
        chart._draw_volume(ax_v, plot_df, colors, vol_lookback)

        direction = "NONE" if has_error else tf_info.get("direction", "NONE")
        setup_info = tf_info.get("setup") if isinstance(tf_info.get("setup"), dict) else {}
        setup_type = "NONE" if has_error else setup_info.get("type", "NONE")
        badge_color = chart.UP if direction == "LONG" else chart.DOWN if direction == "SHORT" else chart.AXIS
        setup_txt = f"  ·  {setup_type}" if setup_type and setup_type != "NONE" else ""
        ax_p.set_title(f"{tf}  ·  {direction}{setup_txt}", color=badge_color,
                        fontsize=title_fs, fontweight="bold", loc="left", pad=6)

        dec = chart.decimals_from_price(float(plot_df["close"].iloc[-1]))
        last_price = chart.format_price(plot_df["close"].iloc[-1], dec)
        ax_p.text(0.99, 0.03, last_price, transform=ax_p.transAxes, color=chart.TEXT,
                   fontsize=price_fs, fontweight="bold", ha="right", va="bottom", zorder=9)

        if has_error:
            ax_p.text(0.5, 0.5, "NO DATA", transform=ax_p.transAxes, color=chart.DOWN,
                       fontsize=title_fs, fontweight="bold", ha="center", va="center")

    header_fs = 20.0 if square else 17.0
    badge_fs = 17.0 if square else 14.0
    footer_fs = 10.0 if square else 7.0
    disclaimer_fs = 9.0 if square else 6.5

    fig.text(0.045, 0.95, f"{symbol}  ·  MULTI-TIMEFRAME", fontsize=header_fs,
              fontweight="bold", color=chart.TEXT, ha="left", va="top")
    ref_df = dfs[valid_tfs[0]]
    chart._draw_change_badge(fig, 0.975, 0.95, chart._calc_24h_change(ref_df), fontsize=badge_fs)
    fig.text(0.045, 0.02, f"BINANCE FUTURES  ·  {symbol}", fontsize=footer_fs,
              color=chart.AXIS, ha="left", va="bottom")
    fig.text(0.98, 0.02,
              "Chart-based analysis for educational purposes only. NOT FINANCIAL ADVICE, DYOR.",
              fontsize=disclaimer_fs, fontweight="bold", color=chart.TEXT, ha="right", va="bottom")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * 2)
    plt.close(fig)
    return out_path
