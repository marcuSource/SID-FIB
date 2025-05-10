//TEAM_AXIS (Equipo 200)

//PLAN 1 (ANAR A LA BANDERA I PATRULLAR) Identificat amb PATROLLING
+flag(F): team(200) 
  <-
  +anar_a_bandera;
  .goto(F).

+target_reached(T): anar_a_bandera & team(200)
  <-
  -anar_a_bandera;
  -target_reached(T);
  ?flag([FX,FY,FZ]); // Posición de la bandera
  
  // Crear 3 puntos en forma de triángulo equilátero alrededor de la bandera a 12 unidades
  P1 = [FX, FY, FZ+12];                                 // Norte
  P2 = [FX+12*0.866, FY, FZ-12*0.5];                    // Sureste (rotación 120º)
  P3 = [FX-12*0.866, FY, FZ-12*0.5];                    // Suroeste (rotación 240º)
  
  // Los ordenamos en sentido antihorario
  C = [P1, P2, P3];
  
  +control_points(C);
  .length(C,L);
  +total_control_points(L);
  +patrullant;
  +patroll_point(0);
  ?control_points(C);
  .nth(0,C,A);
  .goto(A);
  
  .reload; // Colocar el primer ammo pack al llegar a la bandera
  .print("Axis fieldop: Generando el primer ammo pack al llegar a la bandera").

+target_reached(T): patrullant & team(200)
  <-
  .print("Axis fieldop: Punto de patrulla alcanzado");
  ?patroll_point(P);
  -+patroll_point(P+1);
  -target_reached(T).

+patroll_point(P): total_control_points(T) & P<T & team(200)
  <-
  ?control_points(C);
  .nth(P,C,A);
  .reload;
  .goto(A).

+patroll_point(P): total_control_points(T) & P==T & team(200)
  <-
  -patroll_point(P);
  +patroll_point(0).


//-----------------FINAL DEL COMPORTAMENT GENERAL, ARA ACCIONS ESPECÍFIQUES -------------------------

//PLAN 1 PATRULLANT
// Si trobem enemics disparem i bloquejem el plantejament a combatre
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): Health > 0 &  patrullant & team(200)
  <-
  .shoot(5,[X1,Y1,Z1]);
  -patrullant;
  +combatent;
  .print("Axis fieldop: Disparo al enemigo");
  .reload; // Generar ammo porque probablemente necesitemos
  .stop;
  .look_at([X1,Y1,Z1]).


// Si tenim poca vida busquem medikits
+health(H): team(200) & H < 50 & patrullant
  <-
  .print("Axis fieldop: Vida baja, buscando medkit");
  .stop;
  .turn(0.25).


//PLAN 2 COMBATENT

+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): Health > 0 & combatent & team(200)
  <-
  .shoot(5,[X1,Y1,Z1]);
  .print("Axis fieldop: Disparo al enemigo (después de recibir daño)");
  .stop;
  .look_at([X1,Y1,Z1]).

//hem deixat de veure enemics mentre combatiem tornem a patrullar
+combatent: not enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]) & team(200)
  <-
  ?control_points(C);
  -combatent;
  +patrullant;
  .nth(0,C,A);
  .goto(A).

//PLA 3 PERSEGUINT:
//Veiem que algú s'emporta la bandera
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & flag(F) & (F < Position | F > Position) & Type == 1003
  <-
  .print("HE VIST LA BANDERA, la perseguim");
  -patrullant;
  +perseguint_bandera;
  .look_at(Position);
  .goto(Position).

+target_reached(T) : team(200) & perseguint_bandera
  <-
  .print("Hem arribat a la bandera, ja no la tindra l'enemic, patrullo al voltant");
  -target_reached(T);
  -perseguint_bandera;
  +patrullant;
  
  ?position(P);
  .create_control_points(P,5,4,C);
  //Els punts de control passaran a ser els nous
  ?control_points(C);
  .nth(0,C,A);
  .goto(A).

//ACCIONS QUE SON IGUALS PELS 2 PLANS

// Comportamiento de munición
+ammo(A): team(200) & A < 40
  <-
  .print("Axis fieldop: Generando munición para el equipo");
  .reload.

// Ir a por munición si la ve y tiene poca
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & ammo(A) & A < 15 & Type == 1002
  <-
  .print("Axis fieldop: Veo munición y tengo poca, voy a por ella");
  -patrullant;
  +anant_a_municio;
  .goto(Position);
  .look_at(Position).

+target_reached(T): team(200) & anant_a_municio
  <-
  -target_reached(T);
  -anant_a_municio;
  .print("Tenim munició tornem a patrullar");
  +patrullant;
  ?control_points(C);
  .nth(0,C,A);
  .goto(A).
  
// Ir a por medkit si lo ve y tiene poca vida
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & health(X) & X < 60 & Type == 1001 & (patrullant | combatent)
  <-
  .print("Axis fieldop: Veo medkit y tengo poca vida, voy a por él");
  -patrullant;
  -combatent;
  +anant_a_vida;
  .goto(Position);
  .look_at(Position).

+target_reached(T): team(200) & anant_a_vida
  <-
  -target_reached(T);
  -anant_a_vida;
  .print("Tenim Vida tornem a patrullar");
  +patrullant;
  ?control_points(C);
  .nth(0,C,A);
  .goto(A).

// Comportamiento cuando han robado la bandera - perseguir y atacar al portador
+flag_taken: true & team(200) & not hunting_flag_carrier
  <-
  .print("Axis fieldop: Bandera robada, buscando al portador");
  ?flag(F);
  .goto(F);
  -patrullant;
  +hunting_flag_carrier.

// Si ve al portador de la bandera, atacar con prioridad
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): 
    Health > 0 & position([X2,Y2,Z2]) & team(200) & hunting_flag_carrier
  <-
  .shoot(5,[X1,Y1,Z1]); // Disparar con más intensidad
  .print("Axis fieldop: Disparando al posible portador de la bandera");
  .stop;
  .look_at([X1,Y1,Z1]);
  .goto([X1,Y1,Z1]).


//Si veiem un amic i està baix de munició l'anem a recarregar
+friends_in_fov(ID,Type,Angle,Distance,Health,P): team(200) & (combatent | patrullant) & ammo(X) & X < 75
  <-
  .print("Aliat necessita munició, vaig a recarregar-lo");
  +anant_a_recarregar_aliat;
  -combatent;
  -patrullant;
  .look_at(P);
  .goto(P).

+target_reached(T): team(200) & anant_a_recarregar_aliat
  <-
  -target_reached(T);
  -anant_a_recarregar_aliat;
  .reload;
  .print("Hem recarregat a l'aliat, tornem a combatre");
  +combatent; //Si l'aliat necessitava munició probablement era perque estava combatent
  ?control_points(C);
  .nth(0,C,A);
  .goto(A).




//------------------------------------------------------------------------------------------------
////////////////////--------------------ALLIED -----------------------------//////////////////////
//------------------------------------------------------------------------------------------------


  
//TEAM_ALLIED (Equipo 100)

+flag([FX,FY,FZ]): team(100) 
  <-

  ?position([PX,PY,PZ]); // POSICIÓ ACTUAL

  //Posarem 2 ammo entre la bandera i la nostra base per preparar la fugida.

  P1 = [(FX+PX)/3, (FY+PY)/3, (FZ+PZ)/3];                 // primer punt
  
  +primer_ammo(P1);

  +anant_punt_control_1;
  .goto(P1).

//Hem arribat a un dels punts intermitjos i deixarem el ammo
+target_reached(T) : anant_punt_control_1 & team(100)
  <-
  .reload;
  -target_reached(T);
  -anant_punt_control_1;
  +anant_punt_control_2;

  ?flag([FX,FY,FZ]);
  ?position([PX,PY,PZ]);
  
  P2 = [(FX+PX)/2, (FY+PY)/2, (FZ+PZ)/2];                 // segon punt

  +segon_ammo(P2);

  .goto(P2).


+target_reached(T) : anant_punt_control_2 & team(100)
  <-
  .reload;
  -target_reached(T);
  -anant_punt_control_2;
  +anant_a_bandera;

  ?flag(F);

  .print("Anant a la BANDERA");
  .goto(F).


+target_reached(T) : anant_a_bandera & team(100) & not flag_taken
  <-
  .print("Si no tinc jo la bandera algu del meu equip si, tornem a la base que és el que deu estar fent el company");
  -target_reached(T);
  -anant_a_bandera;
  +tornant_amb_bandera;
  ?base(B);
  .goto(B).

//Una altre manera de veure que hem agafat la bandera és amb el flag flag_taken (Ho posem per si agafem la bandera sense estar al pla anant_a_bandera que podria ser per coincidencia)
+flag_taken: team(100)
  <-
  .print("TENIM LA BANDERA, TORNEM A LA BASEE");
  -anant_a_bandera;
  +tornant_amb_bandera;
  ?base(B);
  .look_at(B); //Així ignorem els medikits qeu poden haver-hi a la bandera
  .goto(B).


//Estavem tornant perque algu tenia la bandera i l'estem veient
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1003 & tornant_amb_bandera & not flag_taken & team(100)
  <-
  .print("HE VIST LA BANDERA, la perseguim");
  -tornant_amb_bandera;
  +perseguint_bandera;
  .look_at(Position);
  .goto(Position).

+target_reached(T) : team(100) & perseguint_bandera
  <-
  .print("Hem arribat a la bandera, ja no la tindra el company");
  -target_reached(T);
  -perseguint_bandera;
  +tornant_amb_bandera;
  ?base(B);
  .goto(B).

//-----------------FINAL DEL COMPORTAMENT GENERAL, ARA ACCIONS ESPECÍFIQUES -------------------------
//Cal remarcar que cada acció especifica tindra una resposta o una altre depenent de la fase del pla en la que estiguem (1 posant ammos, 2 anant a per la bandera o 3 tornant a la base)

// ACCIONS DURANT FASE 1 (Posant ammos) (Poc probable que passi res durant aquesta fase així que la podem deixar buida)

// ACCIONS DURANT FASE 2 (Anant a per la bandera) (Sabem que estem en aquesta fase pel fet +anant_a_bandera)

//Dins de les Fases també poden haver-hi sub fases com anar a curar a algú, això a vegades cal incluir-ho en funcions general com disparar.
//Si veiem enemics, tenim la vida molt baixa i estem lluny de la bandera, marxem per recuperar-nos i tornar
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): team(100) & health(X) & X < 30 & anant_a_bandera & flag([FX,FY,FZ]) & (sqrt( (FX-X1)*(FX-X1) + (FZ-Z1)*(FZ-Z1)) > 10.0)
  <-
  .print("Molts enemics, poca vida i molt lluny de la bandera, fem retirada");
  +segon_ammo(P);
  -anant_a_bandera;
  +anant_punt_control_2;  //Així tornem al pla inicial (deixarem mes ammos allà)
  .goto(P).

//Si veiem enemics i estem bé els disparem i seguim avançant cap a la bandera
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): team(100) & Health > 0 & (curant_aliat | anant_a_curarme | anant_a_reload | anant_a_bandera) & ammo(A) & A > 0
  <-
  .print("Enemic detectat, disparem");
  .shoot(5,[X1,Y1,Z1]).


//Si estem baixos de ammo ens tirem ammoS
+ammo(X): team(100) & anant_a_bandera & X < 75
  <-
  .reload.

//SI veig medikits i estic baix de vida vaig a curarme
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1001 & anant_a_bandera & health(X) & X < 75 & team(100)
  <-
  -anant_a_bandera;
  +anant_a_curarme;
  .goto(Position).

+target_reached(T): team(100) & anant_a_curarme
  <-
  -target_reached(T);
  -anant_a_curarme;
  +anant_a_bandera;
  ?flag(F);
  .goto(F).

//Si veig AMMO i no en tinc vaig a buscarla
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1002 & anant_a_bandera & ammo(X) & X == 0 & team(100)
  <-
  -anant_a_bandera;
  +anant_a_reload;
  .goto(Position).

+target_reached(T): team(100) & anant_a_reload
  <-
  -target_reached(T);
  -anant_a_reload;
  +anant_a_bandera;
  ?flag(F);
  .goto(F).

 
// ACCIONS DURANT FASE 3 (Tornant a la Base) (Sabem que estem en aquesta fase pel fet +tornant_amb_bandera)

//SI veig medikits i estic baix de vida vaig a curarme
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1001 & tornant_amb_bandera & health(X) & X < 75 & team(100)
  <-
  -tornant_amb_bandera;
  +anant_a_curarme_tornant;
  .goto(Position).

+target_reached(T): team(100) & anant_a_curarme_tornant
  <-
  -target_reached(T);
  -anant_a_curarme_tornant;
  +tornant_amb_bandera;
  ?base(B);
  .goto(B).