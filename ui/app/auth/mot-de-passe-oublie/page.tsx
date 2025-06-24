"use client";

import { Alert } from "@codegouvfr/react-dsfr/Alert";
import { Button } from "@codegouvfr/react-dsfr/Button";
import { Input } from "@codegouvfr/react-dsfr/Input";
import { Box, Typography } from "@mui/material";
import { Suspense, useState } from "react";
import type { SubmitHandler } from "react-hook-form";
import { useForm } from "react-hook-form";
import type { IGetRoutes, IQuery } from "shared";
import type { IStatus } from "shared/src/routes/_private/auth.routes";

import FormContainer from "@/app/auth/components/FormContainer";
// import Breadcrumb from "@/components/breadcrumb/Breadcrumb";
import { apiGet } from "@/utils/api.utils";

import { PAGES } from "../../../utils/routes.utils";

//@ts-expect-error: TODO fix this
type Route = IGetRoutes["/auth/reset-password"];

const MotDePasseOubliePage = () => {
  const [status, setStatus] = useState<IStatus>();
  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<IQuery<Route>>();

  const onSubmit: SubmitHandler<IQuery<Route>> = async (data) => {
    try {
      //@ts-expect-error: TODO fix this
      await apiGet("/auth/reset-password", {
        querystring: data,
      });

      setStatus({
        error: false,
        message: "Vous allez recevoir un lien vous permettant de réinitialiser votre mot de passe.",
      });
      reset();
    } catch (error) {
      const errorMessage = (error as Record<string, string>)?.message;

      setStatus({
        error: true,
        message:
          errorMessage ??
          "Impossible de réinitialiser votre mot de passe. Vérifiez que vous avez bien saisi votre adresse email.",
      });

      console.error(error);
    }
  };

  return (
    <Suspense>
      {/* <Breadcrumb pages={[PAGES.connexion(), PAGES.motDePasseOublie()]} /> */}
      <FormContainer>
        <Typography variant="h2" gutterBottom>
          Mot de passe oublié
        </Typography>
        <form onSubmit={handleSubmit(onSubmit)}>
          <Input
            label="Votre email"
            state={errors.email ? "error" : "default"}
            stateRelatedMessage={errors.email?.message}
            nativeInputProps={{
              placeholder: "prenom.nom@courriel.fr",
              ...register("email", { required: "Email obligatoire" }),
            }}
          />

          {status?.message && <Alert description={status.message} severity="error" small />}
          <Box mt={2}>
            <Button type="submit">Recevoir un courriel de ré-initialisation</Button>
            <Button
              linkProps={{
                href: PAGES.static.connexion.getPath(),
              }}
              priority="tertiary no outline"
            >
              Annuler
            </Button>
          </Box>
        </form>
      </FormContainer>
    </Suspense>
  );
};
export default MotDePasseOubliePage;
