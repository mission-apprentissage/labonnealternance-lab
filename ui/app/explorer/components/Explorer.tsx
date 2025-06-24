import { Summary } from "@codegouvfr/react-dsfr/Summary";
import { Grid, Typography } from "@mui/material";

import Section from "@/components/section/Section";
import { publicConfig } from "@/config.public";

const anchors = {
  EditeurDuSite: "editeur-du-site",
  DirecteurDeLaPublication: "directeur-de-la-publication",
  HebergementDuSite: "hebergement-du-site",
  Accessibilite: "accessibilite",
  SignalerUnDyfonctionnement: "signaler-un-dyfonctionnement",
};

const summaryData = [
  {
    anchorName: " Section 1",
    anchorLink: anchors.EditeurDuSite,
  },
];

const Explorer = () => {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} lg={3}>
        <Summary
          links={summaryData.map((item) => ({
            linkProps: {
              href: `#${item.anchorLink}`,
            },
            text: item.anchorName,
          }))}
        />
      </Grid>
      <Grid item xs={12} lg={9}>
        <Typography variant="h2" gutterBottom>
          Explorer
        </Typography>

        <Typography>Explorer « {publicConfig.productMeta.productName} »</Typography>

        <Section id={anchors.EditeurDuSite}>
          <Typography variant="h3" gutterBottom>
            Section 1
          </Typography>
          <Typography>Contenu</Typography>
        </Section>
      </Grid>
    </Grid>
  );
};

export default Explorer;
