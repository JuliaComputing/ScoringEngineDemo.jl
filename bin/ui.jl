function ui(model::Model)

    Stipple.page(
        model,
        class="container",
        title="Model Diagnosis",
        head_content=Genie.Assets.favicon_support(),
        [
            heading("Model Diagnosis"),
            row([
                cell(class="st-module col-sm-3",
                    [
                        row(h5("Feature:")),
                        row([
                            Stipple.select(:feature; options=:features)
                            # btn("Report", @click("weave = true"), color="secondary"),
                        ])
                    ]),
                cell(class="st-module col-sm-3",
                    [
                        row(h5("Group Method:")),
                        row([
                            radio(label="Quantiles", fieldname=:groupmethod, val="quantiles", dense=false),
                            radio(label="Linear", fieldname=:groupmethod, val="linear", dense=false),
                        ])
                    ]),
                    cell(class="st-module col-sm-3",
                    [
                        row(h5("New sample:")),
                        row([
                            btn("Report", @click("weave = true"), color="secondary"),
                        ])
                    ]),
            ]),
            # row(h5("One-way effect")),
            row(
                [
                    cell(class="st-module",
                        [
                            plot(:plt_base_trace, layout=:plt_base_layout, config=:plt_base_config)
                        ])
                    cell(class="st-module",
                        [
                            plot(:plot_data, layout=:plot_layout, config=:plot_config)
                        ]
                    )
                ]),
            row([
                cell(class="st-module",
                    [
                        plot(:hist_flux_data, layout=:hist_flux_layout, config=:hist_flux_config)
                    ]),
                cell(class="st-module",
                    [
                        plot(:hist_gbt_data, layout=:hist_gbt_layout, config=:hist_gbt_config)
                    ])
            ]),
        ],
        @iif(:isready)
    )
end