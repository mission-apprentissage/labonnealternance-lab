"use client";

import { Alert } from "@codegouvfr/react-dsfr/Alert";
import { PasswordInput } from "@codegouvfr/react-dsfr/blocks/PasswordInput";
import { Button } from "@codegouvfr/react-dsfr/Button";
import { Input } from "@codegouvfr/react-dsfr/Input";
import { Box } from "@mui/material";
import Typography from "@mui/material/Typography";
import { useRouter } from "next/navigation";
import { Suspense, useState } from "react";
import type { SubmitHandler } from "react-hook-form";
import { useForm } from "react-hook-form";
import type { IBody, IPostRoutes } from "shared";
import type { IStatus } from "shared/src/routes/_private/auth.routes";

import FormContainer from "@/app/auth/components/FormContainer";
import Breadcrumb from "@/components/breadcrumb/Breadcrumb";
import { useAuth } from "@/context/AuthContext";
import { apiPost } from "@/utils/api.utils";

import { PAGES } from "../../../utils/routes.utils";

type Route = IPostRoutes["/_private/auth/login"];

const ConnexionPage = () => {
  const { push } = useRouter();
  const { session, setSession } = useAuth();
  const [status, setStatus] = useState<IStatus>();

  const {
    register,
    handleSubmit,
    formState: { errors },
    //@ts-expect-error: TODO fix this
  } = useForm<IBody<Route>>();
  //@ts-expect-error: TODO fix this
  const onSubmit: SubmitHandler<IBody<Route>> = async (data) => {
    try {
      //@ts-expect-error: TODO fix this
      const session = await apiPost("/_private/auth/login", {
        body: data,
      });
      setSession(session);
    } catch (error) {
      const errorMessage = (error as Record<string, string>)?.message;

      setStatus({
        error: true,
        message: errorMessage ?? "Impossible de se connecter.",
      });

      console.error(error);
    }
  };

  if (session) {
    return push("/");
  }

  return (
    <Suspense>
      <Breadcrumb pages={[PAGES.static.connexion]} />
      <FormContainer>
        <Typography variant="h2" gutterBottom>
          Connectez-vous
        </Typography>
        <form onSubmit={handleSubmit(onSubmit)}>
          <Input
            label="Email (votre identifiant)"
            //@ts-expect-error: TODO fix this
            state={errors.email ? "error" : "default"}
            //@ts-expect-error: TODO fix this
            stateRelatedMessage={errors.email?.message}
            nativeInputProps={{
              placeholder: "prenom.nom@courriel.fr",
              //@ts-expect-error: TODO fix this
              ...register("email", { required: "Email obligatoire" }),
            }}
          />
          <PasswordInput
            label="Mot de passe"
            messagesHint="Mot de passe incorrect"
            messages={
              //@ts-expect-error: TODO fix this
              errors.password?.message
                ? [
                    {
                      //@ts-expect-error: TODO fix this
                      message: errors.password?.message,
                      severity: "error",
                    },
                  ]
                : []
            }
            nativeInputProps={{
              placeholder: "****************",
              //@ts-expect-error: TODO fix this
              ...register("password", {
                required: "Mot de passe obligatoire",
              }),
            }}
          />
          {status?.message && <Alert description={status.message} severity="error" small />}
          <Box mt={2}>
            <Button type="submit">Connexion</Button>
            <Button
              linkProps={{
                href: PAGES.static.motDePasseOublie.getPath(),
              }}
              priority="tertiary no outline"
            >
              Mot de passe oublié
            </Button>
          </Box>
        </form>
      </FormContainer>
    </Suspense>
  );
};
export default ConnexionPage;
