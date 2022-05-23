function ui(model::Model)

    Stipple.page(
        model,
        class="container",
        title="Model Diagnosis",
        head_content=Genie.Assets.favicon_support(),
        [
            heading("Model Diagnosis"),
            row([
                cell(class="st-module",
                    [
                        row(h5("Feature:")),
                        row([
                            Stipple.select(:feature; options=:features)
                        ])
                    ]),
                cell(class="st-module",
                    [
                        row(h5("Group Method:")),
                        row([
                            radio(label="Quantiles", fieldname=:groupmethod, val="quantiles", dense=false),
                            radio(label="Linear", fieldname=:groupmethod, val="linear", dense=false),
                        ])
                    ]),
                cell(class="st-module",
                    [
                        row(h5("New sample:")),
                        row([
                            btn("Resample", @click("resample = true"), color="secondary"),
                        ])
                    ]),
                cell(class="st-module",
                    [
                        row(h5("Another functionality:"))
                    ]),
            ]),
            row(
                [
                    cell(class="st-module",
                        [
                            plot(:one_way_traces, layout=:one_way_layout, config=:one_way_config)
                        ])
                    cell(class="st-module",
                        [
                            plot(:shap_effect_traces, layout=:shap_effect_layout, config=:shap_effect_config)
                        ]
                    )
                ]),
            row([
                cell(class="st-module",
                    [
                        plot(:explain_flux_traces, layout=:explain_flux_layout, config=:explain_flux_config)
                    ]),
                cell(class="st-module",
                    [
                        plot(:explain_gbt_traces, layout=:explain_gbt_layout, config=:explain_gbt_config)
                    ])
            ]),
            row([
                cell(class="st-module",
                    [
                        plot(:hist_flux_traces, layout=:hist_flux_layout, config=:hist_flux_config)
                    ]),
                cell(class="st-module",
                    [
                        plot(:hist_gbt_traces, layout=:hist_gbt_layout, config=:hist_gbt_config)
                    ])
            ]),
        ],
        @iif(:isready)
    )
end