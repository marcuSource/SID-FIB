
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
  
  // Crear 4 puntos en forma de rombo alrededor de la bandera a 10 unidades
  P1 = [FX, FY, FZ+10];    // Norte
  P2 = [FX+10, FY, FZ];    // Este
  P3 = [FX, FY, FZ-10];    // Sur
  P4 = [FX-10, FY, FZ];    // Oeste
  
  // Los ordenamos en sentido antihorario
  C = [P2, P1, P4, P3];
  
  +control_points(C);
  .length(C,L);
  +total_control_points(L);
  +patrullant;
  +patroll_point(0);
  ?control_points(C);
  .nth(0,C,A);
  .goto(A);
  
  .cure. // Colocar el primer medkit al llegar a la bandera

+target_reached(T): patrullant & team(200)
  <-
  .print("Axis medic: Punto de patrulla alcanzado");
  ?patroll_point(P);
  -+patroll_point(P+1);
  -target_reached(T).

+patroll_point(P): total_control_points(T) & P<T & team(200)
  <-
  ?control_points(C);
  .nth(P,C,A);
  .cure;
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
  .print("Axis medic: Disparo al enemigo");
  .cure;//M'imagino que em deuen estar disparant també
  .stop;
  .look_at([X1,Y1,Z1]).


// Si tenim poca vida busquem medikits
+health(H): team(200) & H < 50 & patrullant
  <-
  .print("Axis medic: Vida baja, buscando medkit");
  .stop;
  .turn(0.25).


//PLAN 2 COMBATENT

+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): Health > 0 &  combatent & team(200)
  <-
  .shoot(5,[X1,Y1,Z1]);
  .print("Axis medic: Disparo al enemigo");
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

//Si nomes tinc 20 de vida o menys deixo de combatre i fujo per anar a la bandera on sé que hi ha medikits (Al arribar a la bandera tornaré a la posició on estava)
+health(X): X < 20 & combatent & team(200)
  <-
  -combatent;
  +fujint_per_tornar;
  ?position(P);
  +posicio_abans_de_fujir(P);
  ?flag(F);
  .look_at(F);
  .goto(F).

+target_reached(T): team(200) & fujint_per_tornar
  <-
  -target_reached(T);
  -fujint_per_tornar;
  .print("Tornem a combatre al lloc on erem");
  +combatent;
  ?posicio_abans_de_fujir(P);
  .goto(P);
  .look_at(P).

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
+ammo(A): team(200) & A == 0 
  <-
  .print("Axis medic: Sin munición, buscando más");
  .stop;
  .turn(0.375).

// Ir a por munición si la ve y tiene poca
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & ammo(A) & A < 15 & Type == 1002
  <-
  .print("Axis medic: Veo munición y tengo poca, voy a por ella");
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
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): team(200) & health(X) & X < 60 & Type == 1001  & (patrullant | combatent)
  <-
  .print("Axis medic: Veo medkit y tengo poca vida, voy a por él");
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
  .print("Axis medic: Bandera robada, buscando al portador");
  ?flag(F);
  .goto(F);
  -patrullant;
  +hunting_flag_carrier.

// Si ve al portador de la bandera, atacar con prioridad
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): 
    Health > 0 & position([X2,Y2,Z2]) & team(200) & hunting_flag_carrier
  <-
  .shoot(5,[X1,Y1,Z1]); // Disparar con más intensidad
  .print("Axis medic: Disparando al posible portador de la bandera");
  .stop;
  .look_at([X1,Y1,Z1]);
  .goto([X1,Y1,Z1]).


//Si veiem un amic i esta baix de vida l'anem a curar
+friends_in_fov(ID,Type,Angle,Distance,Health,P): Health < 75 & team(200) & (combatent | patrullant)
  <-
  .print("Aliat baix de vida, vaig a salvarlo");
  +anant_a_curar_aliat;
  -combatent;
  -patrullant;
  .look_at(P);
  .goto(P).

+target_reached(T): team(200) & anant_a_curar_aliat
  <-
  -target_reached(T);
  -anant_a_curar_aliat;
  .cure;
  .print("Hem curat a l'aliat tornem a patrullar");
  +combatent. //Si l'aliat estava baix de vida probablement era perque estava combatent





//------------------------------------------------------------------------------------------------
////////////////////--------------------ALLIED -----------------------------//////////////////////
//------------------------------------------------------------------------------------------------

//TEAM_ALLIED (Equipo 100)

+flag([FX,FY,FZ]): team(100) 
  <-

  ?position([PX,PY,PZ]); // POSICIÓ ACTUAL

  //Posarem 2 ammo entre la bandera i la nostra base per preparar la fugida.

  P1 = [(FX+PX)/3, (FY+PY)/3, (FZ+PZ)/3];                 // primer punt
  
  +primer_medikit(P1);

  +anant_punt_control_1;
  .goto(P1).

//Hem arribat a un dels punts intermitjos i deixarem el medikit
+target_reached(T) : anant_punt_control_1 & team(100)
  <-
  .cure;
  -target_reached(T);
  -anant_punt_control_1;
  +anant_punt_control_2;

  ?flag([FX,FY,FZ]);
  ?position([PX,PY,PZ]);
  
  P2 = [(FX+PX)/2, (FY+PY)/2, (FZ+PZ)/2];                 // segon punt

  +segon_medikit(P2);

  .goto(P2).


+target_reached(T) : anant_punt_control_2 & team(100)
  <-
  .cure;
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
//Cal remarcar que cada acció especifica tindra una resposta o una altre depenent de la fase del pla en la que estiguem (1 posant medikits, 2 anant a per la bandera o 3 tornant a la base)

// ACCIONS DURANT FASE 1 (Posant medikits) (Poc probable que passi res durant aquesta fase així que la podem deixar buida)

// ACCIONS DURANT FASE 2 (Anant a per la bandera) (Sabem que estem en aquesta fase pel fet +anant_a_bandera)

//Dins de les Fases també poden haver-hi sub fases com anar a curar a algú, això a vegades cal incluir-ho en funcions general com disparar.
//Si veiem enemics, tenim la vida molt baixa i estem lluny de la bandera, marxem per recuperar-nos i tornar
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): team(100) & health(X) & X < 30 & anant_a_bandera & flag([FX,FY,FZ]) & (sqrt( (FX-X1)*(FX-X1) + (FZ-Z1)*(FZ-Z1)) > 10.0)
  <-
  .print("Molts enemics, poca vida i molt lluny de la bandera, fem retirada");
  +segon_medikit(P);
  -anant_a_bandera;
  +anant_punt_control_2;  //Així tornem al pla inicial (deixarem mes medikits allà)
  .goto(P).

//Si veiem enemics i estem bé els disparem i seguim avançant cap a la bandera
+enemies_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): Health > 0 & team(100) & (curant_aliat | anant_a_curarme | anant_a_reload | anant_a_bandera) & ammo(A) & A > 0
  <-
  .print("Enemic detectat, disparem");
  .shoot(5,[X1,Y1,Z1]).


//Si veiem un aliat i té poca vida, anem a curar-lo i donar-li support
+friends_in_fov(ID,Type,Angle,Distance,Health,[X1,Y1,Z1]): team(100) & anant_a_bandera & Health < 50
  <-
  .print("Anem a curar a l'aliat",ID);
  -anant_a_bandera; //Ja no anem a bandera, anem a curar a l'aliat
  +curant_aliat;

  .goto([X1,Y1,Z1]).

+target_reached(T): team(100) & curant_aliat
  <-
  .cure;
  -target_reached(T);
  -curant_aliat;
  +anant_a_bandera;
  ?flag(F);
  .goto(F).

//Si estem baixos de vida ens tirem MEDIKITS
+health(H): team(100) & anant_a_bandera & H < 75
  <-
  .cure.

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
+packs_in_fov(ID,Type,Angle,Distance,Health,Position): Type == 1002 & anant_a_bandera & ammo(X) & X == 0
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