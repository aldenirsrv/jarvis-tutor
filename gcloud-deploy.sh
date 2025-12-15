# -- LOCAL EXEC --
chmod 755 ./app.config.sh

# Load application variables
. app.config.sh

# Local variables
echo "APP_VERSION: $APP_VERSION"
echo "APP_NAME: $APP_NAME"
echo "SLUG: $SLUG"
echo "DOCKER_ACCOUNT: $DOCKER_ACCOUNT"
acho "PROJECT_ID: $PROJECT_ID"
gcloud config set project $PROJECT_ID
gcloud config list
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")

# gcloud auth login
# create
# gcloud secrets create firebase-service-account --data-file=configs/firebase/serviceAccountKey.json
# update
# gcloud secrets versions add firebase-service-account --data-file=configs/firebase/serviceAccountKey.json

# gcloud secrets add-iam-policy-binding firebase-service-account \
#   --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
#   --role="roles/secretmanager.secretAccessor"

# gcloud run deploy $APP_NAME-backend --image=docker.io/$DOCKER_ACCOUNT/$APP_NAME-$SLUG-server:$APP_VERSION --region=us-east4 --set-env-vars="FIREBASE_CREDENTIAL_JSON=$(cat ./configs/firebase/serviceAccountKey.json | base64)"

gcloud run deploy $APP_NAME-$SLUG-server \
  --image=docker.io/$DOCKER_ACCOUNT/$APP_NAME-$SLUG-server:$APP_VERSION \
  --region=us-east4 \
  --port=8000 \
  --allow-unauthenticated \
  --timeout=900s \
  --memory=4Gi \
  --cpu=1 \
#   --set-secrets="FIREBASE_CREDENTIAL_JSON=firebase-service-account:latest"

# --startup-probe httpGet.port=8080,httpGet.path=/health,initialDelaySeconds=100,periodSeconds=20,timeoutSeconds=20