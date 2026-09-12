    cfg = load_config(args.config)
    symbol = normalize_symbol(args.symbol, cfg["exchange"]["quote_asset"])
    timeframes = cfg.get("timeframes", ["1h"])

    print(f"[analyze] Independent analysis started for {symbol}")
    result = asyncio.run(analyze_symbol(symbol, cfg))
    text = compose_analysis_text(result)
    print(compose_console_summary(result))

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(OUT_DIR, f"analysis_{result['symbol']}_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[analyze] Finished. Markdown saved to {md_path}")

    # Chart digambar oleh mtfk.py (murni modul penggambar, tidak punya CLI/main
    # sendiri). Import HARUS lazy di sini (bukan di atas file). Meskipun mtfk
    # sekarang juga mengimpor _swing_points secara lazy, chart.py masih
    # mengimpor analyze di level modul -- top-level "from mtfk import ..."
    # di analyze akan memicu circular import (analyze -> mtfk -> chart -> analyze).
    from mtfk import build_mtfk_chart
    chart_path = os.path.join(OUT_DIR, f"{symbol}_multi.png")
    try:
        build_mtfk_chart(
            dfs=result["dfs"],
            symbol=result["symbol"],
            timeframes=timeframes,
            per_tf=result["per_tf"],
            out_path=chart_path,
            cfg=cfg,
        )
        print(f"[analyze] Chart MTF saved to {chart_path}")
    except Exception as exc:
        print(f"[warn] Gagal membuat chart MTF untuk {symbol}: {exc}")

    per_tf = result["per_tf"]
    if per_tf and all("error" in info for info in per_tf.values()):
        print(f"[analyze] All {len(per_tf)} timeframe(s) failed to fetch/analyze. Exiting non-zero.")
        sys.exit(1)


if __name__ == "__main__":
    main()
