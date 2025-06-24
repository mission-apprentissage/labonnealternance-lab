"use client";

import "./profil.css";

import { fr } from "@codegouvfr/react-dsfr";
import { Table } from "@codegouvfr/react-dsfr/Table";
import { Box, Typography } from "@mui/material";
import { styled } from "@mui/material/styles";
import type { TooltipProps } from "@mui/material/Tooltip";
import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
import { useMemo } from "react";

import { ApiKeyAction } from "./components/ApiKeyAction";
import { GenerateApiKey } from "./components/GenerateApiKey";
import { ManageApiKeysBanner } from "./components/ManageApiKeysBanner";
import { useApiKeys, useApiKeysStatut } from "./hooks/useApiKeys";

const CustomWidthTooltip = styled(({ className, ...props }: TooltipProps) => (
  <Tooltip {...props} classes={{ popper: className }} />
))({
  [`& .${tooltipClasses.tooltip}`]: {
    maxWidth: "none",
  },
});

const ProfilPage = () => {
  const apiKeys = useApiKeys();
  const statut = useApiKeysStatut();

  const tableData = useMemo(() => {
    if (apiKeys.isLoading) {
      return [];
    }

    return apiKeys.apiKeys.map((apiKey, index) => {
      const expired = new Date(apiKey.expires_at) < new Date();
      const statut = expired ? "expiré" : "actif";

      return [
        <Typography variant="body1" key="name" className="fr-text--sm">
          {apiKey.name}
        </Typography>,
        <Typography
          variant="body1"
          key="statut"
          className="fr-text--bold"
          color={
            expired
              ? fr.colors.decisions.text.label.pinkTuile.default
              : fr.colors.decisions.artwork.minor.greenBourgeon.default
          }
        >
          <i className={fr.cx(!expired ? "fr-icon-checkbox-circle-fill" : "fr-icon-error-warning-fill")}></i>
          &nbsp;
          {statut}
        </Typography>,
        <Typography variant="body1" key="created_at" className="fr-text--sm">
          {new Date(apiKey.created_at).toLocaleDateString()}
        </Typography>,
        <Typography variant="body1" key="expires_at" className="fr-text--sm">
          {new Date(apiKey.expires_at).toLocaleDateString()}
          <CustomWidthTooltip
            title={
              <Box component="ul" sx={{ margin: fr.spacing("1w") }} className={fr.cx("fr-text--xs")}>
                <li>Tous les jetons d'accès ont une durée de vie de 365 jours ;</li>
                <li>1 mois avant l'expiration, vous serez invite à en créer un nouveau pour prolonger votre usage ;</li>
                <li>
                  Vous avez la possibilité à tout moment de créer un nouveau jeton d'accès avec une nouvelle date
                  d'expiration ;
                </li>
                <li>Il n'est pas possible de prolonger la durée de vie d'un jeton.</li>
              </Box>
            }
            arrow
          >
            <Box
              component={"i"}
              sx={{
                marginLeft: fr.spacing("1v"),
                color: fr.colors.decisions.background.active.blueFrance.default,
              }}
              className={fr.cx("fr-icon-question-line")}
            ></Box>
          </CustomWidthTooltip>
        </Typography>,
        <Typography variant="body1" key="last_used_at" className="fr-text--sm">
          {apiKey.last_used_at ? new Date(apiKey.last_used_at).toLocaleDateString() : "Jamais"}
        </Typography>,
        <ApiKeyAction index={index} key={`action_${index}`} apiKey={apiKey} />,
      ];
    });
  }, [apiKeys]);

  return (
    <Box
      sx={{
        display: "flex",
        gap: fr.spacing("4w"),
        flexDirection: "column",
        marginTop: fr.spacing("9w"),
        marginBottom: fr.spacing("9w"),
        minWidth: fr.breakpoints.values.sm,
      }}
    >
      <Typography variant="h1" color={fr.colors.decisions.text.actionHigh.blueEcume.default}>
        Mon compte
      </Typography>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap" }}>
        <Typography variant="h2" color={fr.colors.decisions.artwork.minor.blueEcume.default}>
          Jetons d’accès API
        </Typography>
      </Box>

      {statut !== "actif-ready" && <GenerateApiKey />}

      <Box>
        <ManageApiKeysBanner key="api-key-banner" />
        {tableData.length > 0 && (
          <Table
            data={tableData}
            fixed
            headers={["Nom", "Statut", "Date de création", "Date d'expiration", "Dernière utilisation", "Action"]}
            style={{
              minWidth: "100%",
              marginBottom: fr.spacing("2w"),
            }}
            className="api-key-table"
          />
        )}
      </Box>

      {statut === "actif-ready" && <GenerateApiKey />}
    </Box>
  );
};

export default ProfilPage;
