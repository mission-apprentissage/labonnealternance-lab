"use client";

import { Alert } from "@codegouvfr/react-dsfr/Alert";
import { PasswordInput } from "@codegouvfr/react-dsfr/blocks/PasswordInput";
import { Button } from "@codegouvfr/react-dsfr/Button";
import { Box, Typography } from "@mui/material";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import type { SubmitHandler } from "react-hook-form";
import { useForm } from "react-hook-form";
import type { IPostRoutes } from "shared";
import type { IStatus } from "shared/src/routes/_private/auth.routes";

import FormContainer from "@/app/auth/components/FormContainer";
// import Breadcrumb from "@/components/breadcrumb/Breadcrumb";
import { apiPost } from "@/utils/api.utils";

import { PAGES } from "../../../utils/routes.utils";
// import { NavLink } from "../../components/NavLink";

//@ts-expect-error: TODO fix this
interface IFormValues extends Zod.input<IPostRoutes["/auth/reset-password"]["body"]> {
  password_confirmation: string;
}

const ModifierMotDePassePage = () => {
  const [status, setStatus] = useState<IStatus>();
  const { push } = useRouter();
  const searchParams = useSearchParams();

  const token = searchParams?.get("passwordToken") ?? "";

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
    watch,
  } = useForm<IFormValues>();

  if (!token) {
    return push(PAGES.static.home.getPath());
  }
  //@ts-expect-error: TODO fix this
  const password = watch("password");
  //@ts-expect-error: TODO fix this
  const onSubmit: SubmitHandler<IFormValues> = async ({ password }) => {
    try {
      //@ts-expect-error: TODO fix this
      await apiPost("/auth/reset-password", {
        headers: {
          authorization: `Bearer ${token}`,
        },
        body: {
          password,
        },
      });

      setStatus({
        error: false,
        message: "Votre mot de passe a bien été modifié",
      });
      reset();

      setTimeout(() => {
        push("/auth/connexion");
      }, 3000);
    } catch (error) {
      const errorMessage = (error as Record<string, string>)?.message;

      setStatus({
        error: true,
        message: errorMessage ?? "Impossible de modifier votre mot de passe.",
      });
      console.error(error);
    }
  };

  return (
    <Suspense>
      {/* <Breadcrumb pages={[PAGES.modifierMotDePasse()]} /> */}
      <FormContainer>
        <Typography variant="h2" gutterBottom>
          {/* {PAGES.modifierMotDePasse().title} */}
        </Typography>
        <form onSubmit={handleSubmit(onSubmit)}>
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
          <PasswordInput
            label="Confirmation de mot de passe"
            messagesHint="Mot de passe incorrect"
            messages={
              errors.password_confirmation?.message
                ? [
                    {
                      message: errors.password_confirmation?.message,
                      severity: "error",
                    },
                  ]
                : []
            }
            nativeInputProps={{
              placeholder: "****************",
              ...register("password_confirmation", {
                required: "Confirmation de mot de passe obligatoire",
                validate: {
                  //@ts-expect-error: TODO fix this
                  match: (value) => value === password || "Les mots de passe ne correspondent pas.",
                },
              }),
            }}
          />
          {status?.message && <Alert description={status.message} severity="error" small />}
          <Box mt={2}>
            <Button type="submit">Modifier mon mot de passe</Button>
          </Box>
        </form>
      </FormContainer>
    </Suspense>
  );
};
export default ModifierMotDePassePage;
